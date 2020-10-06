from torch import nn

from logger.imaging import *
from models.actions import ReinforceGroupWiseAction, SupervisedAction
from models.beta import View
from models.utils import clip_hook
from models.vae import VAE


class GroupVAE(VAE):
    def __init__(self, encoder, decoder, action_encoder, nlatents, nactions, beta, max_capacity=None,
                 capacity_leadin=None, gamma=None, lr=None):
        super().__init__(encoder, decoder, beta=beta, max_capacity=max_capacity, capacity_leadin=capacity_leadin, anneal=1.)
        self.action_encoder = action_encoder
        self.anneal = 1
        self.nlatents = nlatents
        self.nactions = nactions
        self.lr = lr
        self.gamma = gamma if gamma is not None else beta
        self.groups = None

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)

    def unwrap(self, x):
        return self.vae.unwrap(x)

    def rep_fn(self, batch):
        (x, offset), y = batch

        mu, lv = self.vae.unwrap(self.vae.encode(x))
        return mu

    def action_params(self):
        return list(self.action_encoder.parameters())

    def group_params(self):
        return list(self.groups.groups.parameters())

    def vae_params(self):
        return list(self.encoder.parameters()) + list(self.decoder.parameters())

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = super(GroupVAE, self).imaging_cbs(args, logger, model, batch=batch)
        cbs.append(ShowLearntAction(logger, to_tb=True))
        cbs.append(GroupWiseActionPlot(logger, self.groups, self.nlatents, self.nactions, to_tb=True))
        cbs.append(AttentionTb(logger))
        return cbs

    def latent_level_loss(self, z2, mu2, mean=False):
        squares = (z2 - mu2).pow(2)
        if not mean:
            squares = squares.sum() / z2.shape[0]
        else:
            squares = squares.mean()
        return squares

    def recon_level_loss(self, x2_hat, x2, loss_fn, mean=False):
        loss = loss_fn(x2_hat, x2)
        if mean:
            loss = loss / x2[0].numel()
        return loss

    def predict_next_z(self, state):
        raise NotImplementedError

    def loss_fn_predict_next_z(self, state):
        raise NotImplementedError

    def policy_loss(self, state):
        return torch.tensor([0]).to(state['x1'].device), {}, {}

    def main_step(self, batch, batch_nb, loss_fn):
        (x, offset), y = batch
        out = self.vae.main_step((x, y), batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state['lv'], state['z'], state['x_hat']

        mu2, lv2 = self.vae.unwrap(self.vae.encode(y))

        state = {'x1': x, 'x2': y, 'z1': z, 'mu': mu, 'mu2': mu2, 'true_actions': offset}
        z2, outs = self.predict_next_z(state)
        x2_hat = self.vae.decode(z2)

        state.update({
            'z2': z2, 'x2_hat': x2_hat, 'loss_fn': loss_fn
        })

        vae_loss = out['loss']
        pred_loss, loss_logs, loss_out = self.loss_fn_predict_next_z(state)

        out_state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        out_state['recon_hat'] = x2_hat
        out_state['true_recon'] = self.decode(mu2)
        out_state['true_actions'] = offset
        out_state.update(outs)
        out_state.update(loss_out)

        self.global_step += 1

        tensorboard_logs = {
            'metric/loss': vae_loss + pred_loss,
            'metric/pred_loss': pred_loss,
            'metric/total_kl_meaned': self.vae.compute_kl(mu, lv, mean=True),
            'metric/mse_x1': self.recon_level_loss(x_hat, x, loss_fn, mean=True),
            'metric/mse_x2': self.recon_level_loss(x2_hat, y, loss_fn, mean=True),
            'metric/mse_z2': self.latent_level_loss(z2, mu2, mean=True),
            'metric/latent_diff': (z2 - z).pow(2).mean(),
            'metric/mse_z1_mu2': (z - mu2).pow(2).mean()}
        tensorboard_logs.update(loss_logs)
        tensorboard_logs.update(out['out'])
        return {'loss': vae_loss + pred_loss, 'out': tensorboard_logs, 'state': out_state}


class ReinforceGroupVAE(GroupVAE):
    def __init__(self, vae, action_encoder, nlatents, nactions, beta, max_capacity=None,
                 capacity_leadin=None, use_regret=False, base_policy_weight=0.9, base_policy_epsilon=1.0005,
                 group_structure=('c+', 'c-'), num_action_steps=1, multi_action_strategy='reward',
                 reinforce_discount=0.99, gamma=None, use_cob=False, learning_rate=None, entropy_weight=0.):
        """ RGrVAE model

        Args:
            vae (models.VAE): Backbone VAE module
            action_encoder (nn.Module): torch Module that encodes image pairs into a policy distribution
            nlatents (int): Number of latent dimensions
            nactions (int): Number of actions
            beta (float): Weighting for the KL divergence
            max_capacity (float): Max capacity for capactiy annealing
            capacity_leadin (int): Capacity leadin, linearly scale capacity up to max over leadin steps
            use_regret (bool): Use reinforement regret
            base_policy_weight (float): Base weight to apply policy over random
            base_policy_epsilon (float): Increase policy weight by (1-weight)*epsilon
            group_structure (list[str]): Structure of group per latent pair, list of (c+, c-, p, ...)
            num_action_steps (int): Number of actions to allow
            multi_action_strategy (str): One of ['reward', 'returns']
            reinforce_discount (float): Discount rewards factor for calcuating returns
            gamma (float): GrVAE gamma for weighting prediction loss
            use_cob (bool): Allow change of basis for representations
            learning_rate (float): Learning rate
            entropy_weight (float): Exploration entropy weight. Weighted entropy is subtracted from loss
        """
        super().__init__(vae.encoder, vae.decoder, action_encoder, nlatents, nactions, beta, max_capacity,
                         capacity_leadin, gamma, learning_rate)
        self.groups = ReinforceGroupWiseAction(latents=nlatents, action_encoder=action_encoder,
                                               base_policy_weight=base_policy_weight,
                                               base_policy_epsilon=base_policy_epsilon,
                                               use_regret=use_regret, group_structure=group_structure,
                                               decoder=self.decode,
                                               multi_action_strategy=multi_action_strategy,
                                               reinforce_discount=reinforce_discount, use_cob=use_cob,
                                               entropy_weight=entropy_weight)
        self.num_action_steps = num_action_steps
        self.vae = vae

        for p in self.parameters():
            p.register_hook(clip_hook) if p.requires_grad else None

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = super().imaging_cbs(args, logger, model, batch=batch)
        cbs.append(RewardPlot(logger))
        cbs.append(ActionListTbText(logger))
        cbs.append(ActionWiseRewardPlot(logger))
        cbs.append(ActionPredictionPlot(logger))
        cbs.append(ActionDensityPlot(logger))
        cbs.append(ActionStepsToTb(logger))
        return cbs

    def predict_next_z(self, state):
        img_list = [state['x1']]
        z1, x1, x2 = state['z1'], state['x1'], state['x2']
        for i in range(self.num_action_steps):
            z2, out = self.groups.predict_next_z(z1, x1, x2, self.training, state['true_actions'])
            z1 = z2
            x1 = self.decode(z2).detach()
            self.groups.reinforce_params[-1].x = x1
            img_list.append(x1.sigmoid())
        img_list.append(x2)
        out['action_sets'] = img_list
        return z2, out

    def policy_loss(self, state, reset=True):
        return self.groups.loss(state['mu2'], x2=state['x2'], reset=reset, )

    def pred_loss(self, state):
        latent_loss = self.latent_level_loss(state['z2'], state['mu2'], mean=False)
        return latent_loss

    def loss_fn_predict_next_z(self, state, reset=True):
        policy_loss, logs, outs = self.policy_loss(state, reset)
        latent_loss = self.pred_loss(state)
        return policy_loss + latent_loss * self.gamma, logs, outs


class ForwardGroupVAE(GroupVAE):
    def __init__(self, vae, action_encoder, nlatents, nactions, beta, max_capacity=None,
                 capacity_leadin=None, group_structure=('c', 'c', 'p+', 'p-'),
                 num_action_steps=1, gamma=None, learning_rate=None, use_cob=False, ):
        """ Generalised ForwardVAE model

        Args:
            vae (models.VAE): Backbone VAE module
            action_encoder (nn.Module): torch Module that encodes image pairs into a policy distribution
            nlatents (int): Number of latent dimensions
            nactions (int): Number of actions
            beta (float): Weighting for the KL divergence
            max_capacity (float): Max capacity for capactiy annealing
            capacity_leadin (int): Capacity leadin, linearly scale capacity up to max over leadin steps
            group_structure (list[str]): Structure of group per latent pair, list of (c+, c-, p, ...)
            num_action_steps (int): Number of actions to allow
            gamma (float): Gamma for weighting prediction loss
            learning_rate (float): Learning rate
            use_cob (bool): Allow change of basis for representations
        """
        super().__init__(vae.encoder, vae.decoder, action_encoder=action_encoder, nlatents=nlatents, nactions=nactions,
                         beta=beta, max_capacity=max_capacity, capacity_leadin=capacity_leadin, gamma=gamma, lr=learning_rate)
        self.groups = SupervisedAction(nlatents, action_encoder=action_encoder, group_structure=group_structure, use_cob=use_cob)
        self.num_action_steps = num_action_steps
        self.vae = vae

        for p in self.parameters():
            p.register_hook(clip_hook) if p.requires_grad else None

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = super().imaging_cbs(args, logger, model, batch=batch)
        cbs.append(RewardPlot(logger))
        cbs.append(ActionListTbText(logger))
        cbs.append(ActionDensityPlot(logger))
        cbs.append(ActionStepsToTb(logger))
        return cbs

    def predict_next_z(self, state):
        img_list = [state['x1']]
        z1, x1, x2 = state['z1'], state['x1'], state['x2']
        for i in range(self.num_action_steps):
            z2, out = self.groups.predict_next_z(z1, state['true_actions'], self.training)
            z1 = z2
            x1 = self.decode(z2).detach()
            img_list.append(x1.sigmoid())
        img_list.append(x2)
        out['action_sets'] = img_list
        return z2, out

    def pred_loss(self, state):
        latent_loss = self.latent_level_loss(state['z2'], state['mu2'], mean=False)
        return latent_loss

    def loss_fn_predict_next_z(self, state, reset=True):
        latent_loss = self.pred_loss(state)
        return latent_loss * self.gamma, {}, {}


class ActionConvAttentionEncoder(nn.Module):
    def __init__(self, in_latents, in_nc, reps_per_action=4, difference=False, zero_init=True):
        super().__init__()
        self.difference = difference
        actions = in_latents // 2 * reps_per_action
        frame_multi = 1 if difference else 2
        self.net = nn.Sequential(  # 64x64
            nn.Conv2d(in_nc * frame_multi, 32, 3, 2),  # 32x32
            nn.ReLU(True),

            nn.Conv2d(32, 16, 3, 2),  # 16x16
            nn.ReLU(True),

            nn.Conv2d(16, 16, 3, 2),  # 8x8
            nn.ReLU(True),
            View(-1),
            nn.Linear(784, actions)
        )

        for p in self.net.modules():
            if isinstance(p, nn.Conv2d):
                torch.nn.init.orthogonal_(p.weight)

        if zero_init:
            nn.init.ones_(list(self.net.modules())[-1].bias)
            nn.init.zeros_(list(self.net.modules())[-1].weight)

    def forward(self, x):
        if self.difference:
            x = (x[:, 0] - x[:, 1]).unsqueeze(1)
        out = self.net(x.detach()).abs().softmax(-1)

        return out


def rl_group_vae(*ars, **kwargs):
    def _group_vae(args, base_model=None):
        if base_model is not None:
            base_model = base_model
        else:
            from models.models import models
            base_model = models[args.base_model](args)
        action_encoder = ActionConvAttentionEncoder(args.latents, in_nc=args.nc, reps_per_action=len(args.group_structure))
        return ReinforceGroupVAE(base_model, action_encoder,
                                 nlatents=args.latents, nactions=len(args.group_structure) * (args.latents // 2),
                                 beta=args.beta, max_capacity=args.capacity, capacity_leadin=args.capacity_leadin,
                                 use_regret=args.use_regret, base_policy_weight=args.base_policy_weight,
                                 base_policy_epsilon=args.base_policy_epsilon, group_structure=args.group_structure,
                                 num_action_steps=args.num_action_steps,
                                 multi_action_strategy=args.multi_action_strategy,
                                 reinforce_discount=args.reinforce_discount, gamma=args.gvae_gamma,
                                 use_cob=args.use_cob, learning_rate=args.learning_rate,
                                 entropy_weight=args.entropy_weight)

    return _group_vae


def forward_grvae(*ars, **kwargs):
    def fgrvae(args, base_model=None):
        if base_model is not None:
            base_model = base_model
        else:
            from models.models import models
            base_model = models[args.base_model](args)
        action_encoder = SupervisedAction(args.latents, action_encoder=None, group_structure=args.group_structure)
        return ForwardGroupVAE(base_model, action_encoder, nlatents=args.latents,
                               nactions=len(args.group_structure) * (args.latents // 2),
                               beta=args.beta, max_capacity=args.capacity, capacity_leadin=args.capacity_leadin,
                               group_structure=args.group_structure, num_action_steps=args.num_action_steps,
                               gamma=args.gvae_gamma, learning_rate=args.learning_rate, use_cob=args.use_cob)

    return fgrvae
