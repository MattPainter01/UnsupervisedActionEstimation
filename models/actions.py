import copy

from models.group_reps import *
from models.policy_grads import VAEReinforceBase
from models.utils import attn_stats, sprites_label_to_action
from torch import nn


class CoB(nn.Module):
    def __init__(self, subrep, rep_size=4):
        super().__init__()
        self.subrep = subrep
        self.cob = nn.Parameter(torch.eye(rep_size, rep_size).unsqueeze(0).repeat(1, 1, 1), requires_grad=True)

    @property
    def weight(self):
        return self.subrep.weight

    def loss(self, *args, **kwargs):
        return self.subrep.loss(*args, **kwargs) * 1

    def forward(self, x, ac):
        cob = self.cob[ac].squeeze(1)
        cob_inv = cob.inverse()
        y = torch.matmul(cob, x.unsqueeze(-1)).squeeze(-1)
        y2 = self.subrep(y, ac)
        return torch.matmul(cob_inv, y2.unsqueeze(-1)).squeeze(-1)


class GroupWiseAction(nn.Module):
    def __init__(self, latents, action_encoder, group_structure=('c', 'c', 'p+', 'p-'), use_cob=False):
        super().__init__()

        group_labels = {'c': lambda i: CyclicRep(i, rep_size=latents, use_cob=use_cob),
                        'c+': lambda i: CyclicRep(i, rep_size=latents, type=1, use_cob=use_cob),
                        'c-': lambda i: CyclicRep(i, rep_size=latents, type=-1, use_cob=use_cob),
                        'cn': lambda i: CyclicRep(i, rep_size=latents, type='n', use_cob=use_cob),
                        'CN': lambda i: CyclicRep(i, rep_size=latents, type='CN'),
                        '0': lambda i: NoOpRep(i, latents, use_cob=use_cob),
                        'm': lambda i: MatrixRep(i, rep_size=latents, use_cob=use_cob),
                        'm3': lambda i: NdRep(i, rep_size=latents, use_cob=use_cob),
                        'm4': lambda i: NdRep(i, rep_size=latents, use_cob=use_cob, ndim=4),
                        'ddn': lambda i: BasicDnRhoRep(i, rep_size=latents, use_cob=use_cob),
                        }

        groups = []
        for i in range(0, latents // 2):
            for g in group_structure:
                groups.append(group_labels[g](i))

        self._groups = groups
        self.action_encoder = action_encoder
        if action_encoder is not None:
            self.old_action_encoder = copy.deepcopy(action_encoder)
            for p in self.old_action_encoder.parameters():
                p.requires_grad = False

    def to_tb(self, writer, epoch):
        tb_str = '\n\n\n\n'.join([repr(g) for g in self.groups])
        writer.add_text('matrices', tb_str, epoch)

        for i, g in enumerate(self.groups):
            if type(g) == CyclicRep:
                writer.add_scalar('cyclic_orders/order_{}'.format(i), g.get_order(), epoch)
                writer.add_scalar('cyclic_orders/values_{}'.format(i), g.values, epoch)
                writer.add_scalar('cyclic_orders/angles_{}'.format(i), g.values % (2 * math.pi), epoch)

    def forward(self, state):
        return self.predict_next_z(state['z1'], state['x1'], state['x2'])

    def get_attn(self, x1, x2, prev_step=False):
        img_pair = torch.cat([x1, x2], 1)
        if prev_step:
            return self.old_action_encoder(img_pair)
        else:
            return self.action_encoder(img_pair)

    def update_prev_params(self):
        self.old_action_encoder.load_state_dict(self.action_encoder.state_dict())
        map(lambda p: p.detach(), self.old_action_encoder.parameters())
        map(lambda p: p.requires_grad(False), self.old_action_encoder.parameters())

    def predict_next_z(self, z1, x1, x2):
        raise NotImplementedError

    def next_rep(self, z, ac):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        return 0, {}, {}


class ReinforceGroupWiseAction(GroupWiseAction, VAEReinforceBase):
    def __init__(self, latents=4, action_encoder=None, base_policy_weight=0.9, base_policy_epsilon=1.0005,
                 normalised_reward=True, use_regret=False,
                 group_structure=('c', 'c', 'p+', 'p-'), decoder=None,
                 multi_action_strategy='reward', reinforce_discount=0.99, use_cob=False,
                 entropy_weight=0.):
        GroupWiseAction.__init__(self, latents, action_encoder=action_encoder, group_structure=group_structure,
                                 use_cob=use_cob)
        VAEReinforceBase.__init__(self, base_policy_weight=base_policy_weight, base_policy_epsilon=base_policy_epsilon,
                                  normalised_reward=normalised_reward, use_regret=use_regret, decoder=decoder,
                                  rep_fn=self.next_rep, multi_action_strategy=multi_action_strategy,
                                  reinforce_discount=reinforce_discount, entropy_weight=entropy_weight)
        self.groups = GroupApply(self._groups)

    def predict_next_z(self, z1, x1, x2, training=True, true_actions=None):
        attn = self.get_attn(x1, x2)

        z2 = self.sample_next_z(attn, z1, training=training)
        try:
            out = attn_stats(attn, true_actions)
        except Exception as e:
            print(e)
            out = {}
        return z2, out

    def next_rep(self, z, ac, orders=None):
        return self.groups(z, ac, orders=orders)

    def loss(self, real, x2, reset=True):
        self.update_prev_params()
        policy_loss, tb_dict, out = VAEReinforceBase.loss(self, real, x2)

        loss = 0
        for g in self.groups:
            loss = loss + g.loss(real, x2)

        tb_dict['metric/group_loss'] = loss.item()
        super().reset() if reset else []
        return policy_loss + loss, tb_dict, out


class SupervisedAction(GroupWiseAction):
    def __init__(self, latents=4, action_encoder=None, group_structure=('c', 'c', 'p+', 'p-'), use_cob=True):
        GroupWiseAction.__init__(self, latents, action_encoder=action_encoder, group_structure=group_structure,
                                 use_cob=use_cob)
        self.groups = GroupApply(self._groups)

    def predict_next_z(self, z1, ac, training=True, true_actions=None):
        ac = sprites_label_to_action(ac).view(z1.shape[0])

        z2 = self.groups(z1, ac)
        return z2, {}

    def next_rep(self, z, ac, orders=None):
        ac = sprites_label_to_action(ac).view(z.shape[0])

        return self.groups(z, ac, orders=orders)

    def loss(self, real, x2, reset=True):
        loss = 0
        for g in self.groups:
            loss = loss + g.loss(real, x2)

        super().reset() if reset else []
        return loss, {}, {}
