"""
Code adapted from https://github.com/Caselles/NeurIPS19-SBDRL

There is also a more general implementation in /models/group_vae: ForwardGroupVAE which uses the same framework
we use for RGrVAE and allows any number of latents and group structures.
"""

import torch.nn.functional as F
from torch import nn

from logger.imaging import *
from models.utils import ParallelActions
from models.vae import VAE


def weight_hook(up):
    def hook_fn(grad):
        new_grad = grad.clone()
        if up:
            new_grad[0][2:] = torch.Tensor([0., 0.]).to(grad.device)
            new_grad[1][2:] = torch.Tensor([0., 0.]).to(grad.device)
            new_grad[2] = torch.Tensor([0., 0., 0., 0.]).to(grad.device)
            new_grad[3] = torch.Tensor([0., 0., 0., 0.]).to(grad.device)
        else:
            new_grad[0] = torch.Tensor([0., 0., 0., 0.]).to(grad.device)
            new_grad[1] = torch.Tensor([0., 0., 0., 0.]).to(grad.device)
            new_grad[2][:2] = torch.Tensor([0., 0.]).to(grad.device)
            new_grad[3][:2] = torch.Tensor([0., 0.]).to(grad.device)
        return new_grad

    return hook_fn


def bias_hook():
    def hook_fn(grad):
        return torch.Tensor([0., 0., 0., 0.]).to(grad.device)

    return hook_fn


class GroupWrapper:
    def __init__(self, group_list):
        self.groups = group_list

    def to_tb(self, writer, current_epoch):
        from tabulate import tabulate
        weight_mats = [g.weight for g in self.groups]
        reprs = [tabulate(w.cpu().numpy()).replace('\n', '\n\n') for w in weight_mats]

        tb_str = '\n\n\n\n'.join([r for r in reprs])
        writer.add_text('matrices', tb_str, current_epoch)


class ForwardEncoder(nn.Module):
    def __init__(self, Z_DIM, complexity=1, nc=1):
        super().__init__()

        layers_multi = [1, 1]
        if complexity > 1:
            layers_multi = [2, 2]

        self.conv1 = nn.Conv2d(nc, 32, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32*layers_multi[0], 4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32*layers_multi[0], 32*layers_multi[1], 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32*layers_multi[1], 32, 4, stride=2, padding=1)

        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 2 * Z_DIM)

        # torch.nn.init.zeros_(self.fc3.weight)
        # torch.nn.init.ones_(self.fc3.bias)

    def forward(self, x):
        h = F.selu(self.conv1(x))
        h = F.selu(self.conv2(h))
        h = F.selu(self.conv3(h))
        h = F.selu(self.conv4(h))
        h = h.view(-1, h.shape[1] * h.shape[2] * h.shape[3])
        h = F.selu(self.fc1(h))
        h = F.selu(self.fc2(h))
        h = F.selu(self.fc3(h))
        return h


class ForwardDecoder(nn.Module):
    def __init__(self, Z_DIM, complexity=1, nc=1):
        super().__init__()
        self.fc4 = nn.Linear(Z_DIM, 256)
        self.fc5 = nn.Linear(256, 512)

        layers_multi = [1, 1]
        if complexity == 2:
            layers_multi = [2, 2]
        if complexity == 3:
            layers_multi = [4, 2]
        self.deconv1 = nn.ConvTranspose2d(32, 32*layers_multi[0], 4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(32*layers_multi[0], 32*layers_multi[1], 4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(32*layers_multi[1], 32, 4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, nc, 4, stride=2, padding=1)

        # torch.nn.init.zeros_(self.deconv4.weight)
        # torch.nn.init.ones_(self.deconv4.bias)

    def forward(self, z):
        h = F.selu(self.fc4(z))
        h = F.selu(self.fc5(h).reshape(-1, 32, 4, 4))
        h = F.selu(self.deconv1(h))
        h = F.selu(self.deconv2(h))
        h = F.selu(self.deconv3(h))
        h = self.deconv4(h)
        return h


class ForwardVAE(nn.Module):
    def __init__(self, Z_DIM=4, beta=1, pred_z_loss_type='latent', max_capacity=None, capacity_leadin=None, nc=1):
        super(ForwardVAE, self).__init__()
        self.Z_DIM = Z_DIM
        self.beta = beta
        self.pred_z_loss_type = pred_z_loss_type
        self.nlatents = Z_DIM
        self.nactions = 4
        self.anneal = 1
        self.anneal_eps = 0.995
        self.capacity = max_capacity
        self.capacity_leadin = capacity_leadin
        self.global_step = 0

        # Encoder layers
        self.encoder = ForwardEncoder(Z_DIM, nc=nc)

        # Decoder layers
        self.decoder = ForwardDecoder(Z_DIM, nc=nc)

        # Physics layers
        self.A_1 = nn.Linear(Z_DIM, Z_DIM)
        self.A_2 = nn.Linear(Z_DIM, Z_DIM)
        self.A_3 = nn.Linear(Z_DIM, Z_DIM)
        self.A_4 = nn.Linear(Z_DIM, Z_DIM)

        self.A_1.weight = torch.nn.Parameter(torch.Tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        self.A_2.weight = torch.nn.Parameter(torch.Tensor([[1, 1, 0, 0], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]))
        self.A_3.weight = torch.nn.Parameter(torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
        self.A_4.weight = torch.nn.Parameter(torch.Tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 1, 1]]))
        self.A_1.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0]))
        self.A_2.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0]))
        self.A_3.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0]))
        self.A_4.bias = torch.nn.Parameter(torch.Tensor([0, 0, 0, 0]))

        self.A_1.weight.register_hook(weight_hook(True))
        self.A_2.weight.register_hook(weight_hook(True))
        self.A_3.weight.register_hook(weight_hook(False))
        self.A_4.weight.register_hook(weight_hook(False))

        self.A_1.bias.register_hook(bias_hook())
        self.A_2.bias.register_hook(bias_hook())
        self.A_3.bias.register_hook(bias_hook())
        self.A_4.bias.register_hook(bias_hook())

        self.action_mapping = [
            self.A_1, self.A_3, self.A_2, self.A_4
        ]

        self.groups = GroupWrapper(self.action_mapping)

    def next_rep(self, z, action, cuda=True):
        state = {'z': z, 'action': [action, ], 'cuda': cuda}
        return self.predict_next_z(state)[0]

    def encode(self, x):
        return self.encoder(x)

    def unwrap(self, x):
        return torch.split(x, x.shape[1] // 2, dim=1)

    def reparameterize(self, mu_and_logvar):
        mu, logvar = self.unwrap(mu_and_logvar)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)

    def predict_next_z(self, state):
        z, action = state['z'], state['action']

        res = torch.tensor([]).to(z.device)

        for i, ac in enumerate(action):
            A = self.action_mapping[ac]
            z_plus_1 = A(z[i])
            res = torch.cat((res, z_plus_1.reshape(1, self.Z_DIM)), dim=0).reshape(-1, self.Z_DIM)

        return res, {}

    def loss_fn_predict_next_z(self, state):

        latent_loss = 0
        if self.pred_z_loss_type in ['latent', 'both']:
            latent_loss += self.latent_level_loss(state, mean=True)
        if self.pred_z_loss_type in ['recon', 'both']:
            latent_loss += self.recon_level_loss(state, mean=True)

        return latent_loss, {}, {}

    def train_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def val_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def control_capacity(self, total_kl, global_step):
        if self.capacity is not None:
            leadin = 1e5 if self.capacity_leadin is None else self.capacity_leadin
            delta = torch.tensor((self.capacity / leadin) * global_step)
            return (total_kl - delta).clamp(min=0) * self.beta
        else:
            return total_kl*self.beta

    def vae_recon_loss(self, recon_x, true_x):
        return F.mse_loss(recon_x.sigmoid(), true_x, reduction='none').mean()

    def divergence_loss(self, logvar, mu):
        return -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()))

    def main_step(self, batch, batch_nb, loss_fn):

        (frames, actions), target_batch = batch

        (recon_x, mu_and_logvar, z_plus_1, z, targets), state = self.forward(frames, actions, target_batch)

        mu, logvar = self.unwrap(mu_and_logvar)

        # loss_recon = loss_fn(recon_x, frames).mean()
        loss_recon = self.vae_recon_loss(recon_x, frames)
        kld = self.divergence_loss(logvar, mu)
        vae_loss = (1 * loss_recon) + (self.anneal * kld * self.control_capacity(kld, self.global_step))

        state.update({
            'pred': z_plus_1, 'z2': z_plus_1, 'x2_hat': self.decode(z_plus_1),
            'loss_fn': lambda x, p: F.binary_cross_entropy_with_logits(x, p, reduction='none').mean(),
            'real': targets,
        })
        state['recon'] = state['x2_hat']

        loss_predict_next_z, loss_logs, loss_out = self.loss_fn_predict_next_z(state)

        loss_total = vae_loss + loss_predict_next_z
        self.anneal = self.anneal * self.anneal_eps

        if batch_nb == 0:
            try:
                self.generate_reconstructed_data()
            except:
                warnings.warn('Failed to generate reconstructed data')

        state.update(loss_out)
        tensorboard_logs = {'metric/recon_loss': loss_recon,
                            'metric/total_kl': -0.5 * torch.mean((1 + logvar - mu.pow(2) - logvar.exp()).sum(-1)),
                            'forward/predict': loss_predict_next_z,
                            'metric/loss': loss_total,
                            'forward/anneal': torch.tensor(self.anneal).float(),
                            'metric/mse_x2': self.recon_level_loss(state, mean=True),
                            'metric/mse_z2': self.latent_level_loss(state, mean=True),
                            'metric/latent_diff': (z_plus_1 - z).pow(2).mean(),
                            'metric/mse_z1_mu2': (z - state['mut']).pow(2).mean()
        }
        tensorboard_logs.update(loss_logs)
        self.global_step += 1
        out = {
            'loss': loss_total,
            'out': tensorboard_logs,
            'state': state
        }

        if not self.training and batch_nb == 0:
            dets, dets_mean, angle_mean, angle_std, angles = self.get_matrix_dets()
            out['out'].update({'forward/angle_{}'.format(i): a for i, a in enumerate(angles)})
            out['out'].update({
                    'forward/dets_mean': dets_mean,
                    'forward/mean_angle': angle_mean,
                    'forward/angle_std': angle_std,
                })

        return out

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = [
            ShowRecon(), ReconToTb(logger),
            LatentWalk(logger, args.latents, list(range(args.latents)), input_batch=batch, to_tb=True),
            ShowLearntAction(logger, to_tb=True),
            GroupWiseActionPlot(logger, self, self.nlatents, self.nactions, to_tb=True)
        ]
        return cbs

    def rep_fn(self, batch):
        (frames, actions), target_batch = batch
        (recon_x, mu_and_logvar, z_plus_1, z, targets), state = self.forward(frames, actions, target_batch)
        return self.unwrap(mu_and_logvar)[0]

    def forward(self, x, action, target_batch, encode=False, mean=False, decode=False):

        if decode:
            return self.decode(x)
        mu_and_logvar = self.encode(x)
        mut_and_logvart = self.encode(target_batch)
        mut = self.unwrap(mut_and_logvart)[0]
        z = self.reparameterize(mu_and_logvar)

        input_state = {
            'z': z, 'action': action, 'x': x, 'target': target_batch, 'mut': mut, 'x2': target_batch,
            'true_actions': action,
        }
        z_plus_1, state_stats = self.predict_next_z(input_state)

        mu, lv = self.unwrap(mu_and_logvar)
        if encode:
            if mean:
                return mu
            return z, z_plus_1

        x_hat = self.decode(z)
        recon = x_hat
        true_recon = self.decode(mut)

        state = {'x': x, 'y': x, 'x_hat': x_hat, 'recon': recon, 'mu': mu, 'lv': lv, 'mut': mut,
                 'x1': x, 'x2': target_batch, 'recon_hat': self.decode(z_plus_1), 'true_recon': true_recon}
        state.update(state_stats)
        return (x_hat, mu_and_logvar, z_plus_1, z, mut), state

    def latent_level_loss(self, state, mean=False):
        z2, mu2 = state['z2'], state['mut']
        squares = (z2 - mu2).pow(2)
        if not mean:
            squares = squares.sum() / z2.shape[0]
        else:
            squares = squares.mean()
        return squares

    def recon_level_loss(self, state, mean=False):
        x2_hat, x2, loss_fn = state['x2_hat'], state['x2'], state['loss_fn']
        loss = loss_fn(x2_hat, x2)
        return loss

    def get_matrix_dets(self):
        import math

        A_1 = np.array(self.A_1.weight.cpu().detach())
        A_2 = np.array(self.A_2.weight.cpu().detach())

        A_3 = np.array(self.A_3.weight.cpu().detach())
        A_4 = np.array(self.A_4.weight.cpu().detach())

        rot_A_1 = np.array([A_1[0][:2], A_1[1][:2]])
        rot_A_2 = np.array([A_2[0][:2], A_2[1][:2]])

        rot_A_3 = np.array([A_3[2][2:], A_3[3][2:]])
        rot_A_4 = np.array([A_4[2][2:], A_4[3][2:]])

        rot_matrices = [rot_A_1, rot_A_2, rot_A_3, rot_A_4]
        angle_mean = np.zeros(1)
        angle_std = np.zeros(1)

        try:
            mangles = []
            for m in rot_matrices:
                angles = [math.acos(m[0][0]), math.asin(m[0][1]), math.asin(m[1][0]), math.acos(m[1][1])]
                # angles = [m[0][0], m[0][1], m[1][0], m[1][1]]
                mangles.append(np.array([abs(a) for a in angles]).mean())

            zero_zero = np.array([abs(math.acos(m[0][0])) for m in rot_matrices])
            zero_one = np.array([abs(math.asin(m[0][1])) for m in rot_matrices])
            one_zero = np.array([abs(math.asin(m[1][0])) for m in rot_matrices])
            one_one = np.array([abs(math.acos(m[1][1])) for m in rot_matrices])
            angle_mean = np.concatenate([zero_zero, zero_one, one_zero, one_one]).mean()
            angle_std = np.concatenate([zero_zero, zero_one, one_zero, one_one]).std()

        except:
            pass

        dets = []
        for mat in rot_matrices:
            dets.append(np.linalg.det(mat))
        return torch.tensor(dets), torch.tensor(dets).mean(), torch.tensor(angle_mean), torch.tensor(
            angle_std), torch.tensor(mangles)

    def generate_reconstructed_data(self):
        """ Should work """
        z = torch.tensor([1., 1., 1., 1.]).cuda()
        # im = self.forward(z, action=None, target_batch=z, decode=True).sigmoid().detach().cpu().numpy().reshape(-1, 3, 64, 64).transpose((0, 2, 3, 1))
        ims = []
        for ac in [-1, 0, 1, 2]:
            aux = []
            z = torch.tensor([1., 1., 1., 1.]).cuda()
            for i, action in enumerate(torch.Tensor(np.ones(15)) + ac):
                im = self.forward(z.cuda(), action=None, target_batch=z.cuda(),
                                  decode=True).sigmoid().detach().cpu().numpy().reshape(-1, 1, 64, 64).transpose(
                    (0, 2, 3, 1))
                im = im.reshape(64, 64)
                aux.append(im)
                action = action.long().cuda()
                next_z = self.next_rep(z.unsqueeze(0), action.view(1, 1), cuda=True)
                # next_z = vae.predict_next_z(z.unsqueeze(0),action.view(1,1), cuda=False)
                z = next_z
            ims.append(aux)

        import matplotlib.pyplot as plt
        plt.close()
        fig, ax = plt.subplots(nrows=4, ncols=15, figsize=(15, 4))
        # fig.subplots_adjust(left=0.125, right=0.9, bottom=0.25, top=0.75, wspace=0.1, hspace=0.1)
        for k, i in enumerate(ax):
            for j, axis in enumerate(i):
                axis.axis('off')
                axis.imshow(ims[k][j])
                axis.set_xticklabels([])
                axis.set_yticklabels([])
                # axis.set_aspect(1)
        plt.tight_layout()
        plt.savefig('./images/reconstruction_again.png')
        return

    def linear_interpolation(self, image_origin, image_destination, number_frames):
        """ Not Tested"""

        res = []
        res.append(image_origin.reshape(1, 3, 64, 64))

        origin_z = self.forward(np.array(image_origin).reshape((1, 3, 64, 64)), encode=True)[0]
        final_z = self.forward(np.array(image_destination).reshape((1, 3, 64, 64)), encode=True)[0]

        for i in range(0, number_frames + 1):
            i /= number_frames
            print(i)
            translat_img = ((1 - i) * origin_z) + (i * final_z)
            res.append(self.forward(np.array(translat_img), decode=True)[0])

        res.append(image_destination.reshape(1, 3, 64, 64))

        return np.array(res)


class BetaForward(VAE):
    def __init__(self, args):
        complexity = 1
        try:
            if args.noise_name == 'BG':
                complexity = 3
            if args.noise_name in ['Salt', 'Gaussian']:
                complexity = 2
        except:
            warnings.warn('Could not find the noise type')

        super().__init__(ForwardEncoder(args.latents, complexity, nc=args.nc), ForwardDecoder(args.latents, complexity, nc=args.nc), args.beta, args.capacity, args.capacity_leadin)


def forward_vae(args):
    return ForwardVAE(args.latents, args.beta, args.pred_z_loss_type, args.capacity, args.capacity_leadin, nc=args.nc)


def beta_forward(args):
    return BetaForward(args)
