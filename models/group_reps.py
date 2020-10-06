import math

import torch
from torch import nn

from models.utils import ParallelActions


class GRep(nn.Module):
    def __init__(self, action, rep_size, use_cob=True):
        super().__init__()
        self.action = action
        self.rep_size = rep_size
        self.use_cob = use_cob
        if use_cob:
            self.cob = torch.nn.Parameter(torch.eye(rep_size, rep_size).unsqueeze(0).repeat(1, 1, 1), requires_grad=True)
        else:
            self.cob = torch.nn.Parameter(torch.tensor([]), requires_grad=False)

    def det(self):
        return 0.

    def cyclic_angle(self):
        return 0.

    @property
    def weight(self):
        if self.use_cob:
            cob = self.cob
            cob_inv = self.cob.inverse()
            return torch.matmul(torch.matmul(cob_inv, self.rep()), cob).squeeze(0)
        else:
            return self.rep()

    def cob_loss(self):
        return (torch.norm(self.cob, p=2, dim=1, keepdim=False) - 1).abs().sum()

    def rep(self, orders=None):
        pass

    def loss(self, real, x2):
        return torch.tensor([0], device=real.device)

    def reset(self):
        pass


class NoOpRep(GRep):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer('r', torch.eye(self.rep_size, self.rep_size, requires_grad=False))

    def rep(self, orders=None):
        return self.r

    def forward(self, x):
        return x


class MatrixRep(GRep):
    def __init__(self, action, rep_size, use_cob=False):
        super().__init__(action, rep_size, use_cob)
        self.values = torch.nn.Parameter(torch.ones(2, 2).unsqueeze(0), requires_grad=True)

    def __repr__(self):
        return "Action: {} - Det: {} - Angle: {} - M: {}".format(self.action, self.det(), self.cyclic_angle(), str(self.values.data))

    def det(self):
        return torch.det(self.values)

    def cyclic_angle(self):
        m = self.values[0]
        angles = [torch.acos(m[0,0]), torch.asin(m[0,1]), torch.acos(m[1, 1]), torch.asin(m[1, 0])]
        angles = torch.stack([a.abs() for a in angles])
        return angles.mean(), angles.std()

    def mat_to_rep(self, mat):
        base_rep = torch.eye(self.rep_size, self.rep_size, device=self.values.device)
        base_rep[self.action*2:self.action*2+2, self.action*2:self.action*2+2] = mat
        return base_rep.unsqueeze(0)

    def id_loss(self):
        return (self.values - torch.eye(2, 2, device=self.values.device)).sum()

    def loss(self, *args, **kwargs):
        det = torch.det(self.values)
        cobloss = self.cob_loss() if self.use_cob else det*0
        return abs(det-1) * 1 + cobloss*1# + self.id_loss()*0.1

    def rep(self, orders=None):
        mat = self.mat_to_rep(self.values).squeeze(0)
        if orders is not None:
            mat = mat.pow(orders+1e-6)
        return mat

    def forward(self, z):
        rep = self.mat_to_rep(self.values).unsqueeze(0)
        return torch.matmul(rep, z.unsqueeze(-1)).squeeze(-1)


class NdRep(GRep):
    def __init__(self, action, rep_size, use_cob=False, ndim=3):
        super().__init__(action, rep_size, use_cob)
        self.ndim = ndim
        self.values = torch.nn.Parameter(torch.ones(ndim, ndim).unsqueeze(0), requires_grad=True)

    def __repr__(self):
        return "Action: {} - Det: {} - Angle: {} - M: {}".format(self.action, self.det(), self.cyclic_angle(), str(self.values.data))

    def det(self):
        return torch.det(self.values)

    def mat_to_rep(self, mat):
        base_rep = torch.eye(self.rep_size, self.rep_size, device=self.values.device)
        base_rep[0:self.ndim, 0:self.ndim] = mat
        return base_rep.unsqueeze(0)

    def id_loss(self):
        return (self.values - torch.eye(self.ndim, self.ndim, device=self.values.device)).sum()

    def loss(self, *args, **kwargs):
        det = torch.det(self.values)
        cobloss = self.cob_loss() if self.use_cob else det*0
        return abs(det-1) * 0 + cobloss*1# + self.id_loss()*0.1

    def rep(self, orders=None):
        mat = self.mat_to_rep(self.values).squeeze(0)
        if orders is not None:
            mat = mat.pow(orders+1e-6)
        return mat

    def forward(self, z):
        rep = self.mat_to_rep(self.values).unsqueeze(0)
        return torch.matmul(rep, z.unsqueeze(-1)).squeeze(-1)


class CyclicRep(GRep):
    def __init__(self, action, rep_size=4, type=0, use_cob=True):
        super().__init__(action, rep_size, use_cob)
        self.type = type
        if type == 'n' or type == 'CN':
            self.values = torch.nn.Parameter(torch.rand(1, 1).unsqueeze(0), requires_grad=True)
        else:
            self.values = torch.nn.Parameter(torch.rand(1, 1).unsqueeze(0)*type, requires_grad=True)

    def det(self):
        return 1.

    def cyclic_angle(self):
        return self.values

    def loss(self, real, x2):
        if self.use_cob:
            return self.values.abs()*0.1 + self.cob_loss()*1
        else:
            return self.values.abs() * 0.0

    def get_order(self):
        v = self.values.data.cpu().numpy()[0]
        return (2*math.pi/v)[0]

    def __repr__(self):
        if self.use_cob:
            v = self.values.data.cpu().numpy()[0]
            repr_str = "Action: {} - {} - C{} - cob{} \n\n".format(self.action, v, str(self.get_order()), str(self.weight))
            return repr_str
        else:
            v = self.values.data.cpu().numpy()[0]
            repr_str = "Action: {} - {} - C{} \n\n".format(self.action, v, str(self.get_order()))
            return repr_str

    def value_to_rot(self, values):
        c, s = torch.cos(values), torch.sin(values)
        return torch.stack([
            c, -s,
            s, c
        ], 0).view(-1, 2, 2)

    def rot_to_rep(self, rot):
        base_rep = torch.eye(self.rep_size, self.rep_size, device=self.values.device)
        base_rep[self.action*2:self.action*2+2, self.action*2:self.action*2+2] = rot
        return base_rep.unsqueeze(0)

    def __pow__(self, powers):
        if powers is None: powers = 1
        angles = self.values * powers
        ret = self.rot_to_rep(self.value_to_rot(angles))
        return ret

    def rep(self, powers=1):
        rep = self ** powers
        return rep.squeeze(0)

    def forward(self, z, powers=1):
        rep = self.rep(powers)
        return torch.matmul(rep.unsqueeze(0), z.unsqueeze(-1)).squeeze(-1)


class BasicDnRhoRep(GRep):
    def __init__(self, action, rep_size=4, use_cob=False):
        super().__init__(action, rep_size, use_cob=use_cob)

    def __repr__(self):
        repr_str = "Action: Dn \n\n"
        return repr_str

    def rep(self):
        base_rep = torch.eye(self.rep_size, self.rep_size, device='cuda')

        rep = torch.tensor([
            [0, 1],
            [1, 0]
        ]).cuda()
        base_rep[self.action*2:self.action*2+2, self.action*2:self.action*2+2] = rep
        return base_rep

    def forward(self, z):
        rep = self.rep()
        return torch.matmul(rep, z.unsqueeze(-1)).squeeze(-1)


class GroupApply(nn.Module):
    def __init__(self, group_list):
        super().__init__()
        self.group_list = torch.nn.ModuleList(group_list)

    def __len__(self):
        return len(self.group_list)

    def forward(self, z, ac, orders=None):
        out = ParallelActions(self.group_list)(z, ac, orders=orders)
        return out

    def __iter__(self):
        return iter(self.group_list)
