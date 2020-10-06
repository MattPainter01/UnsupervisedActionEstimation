import torch
from torch.distributions import Categorical


class ReinforceParams:
    def __init__(self, probs, z, z2, actions, attn_dist):
        self.probs = probs
        self.x = None
        self.z = z
        self.z2 = z2
        self.actions = actions
        self.attn_dist = attn_dist
        self.cat_dist = Categorical(attn_dist)
        self.reward = None
        self.pure_rewards = None
        self.policy_loss = None
        self.regret = None
        self.returns = None
        self.returns_loss = None
        self.entropy_ratio = None
        self.values = None
        self.batch_per_action_rewards = None

    def clone(self):
        p = ReinforceParams(self.probs, self.z, self.z2, self.actions, self.attn_dist)
        p.x = self.x
        return p


class ReinforceBase:
    def __init__(self, base_policy_weight=1., base_policy_epsilon=0.000, normalised_reward=False,
                 reinforce_discount=0.9, use_regret=False):
        super().__init__()
        self.policy_weight = base_policy_weight
        self.policy_epsilon = base_policy_epsilon
        self.reinforce_params = []
        self.normalised_reward = normalised_reward
        self.reinforce_discount = reinforce_discount
        self.use_regret = use_regret

    def reinforce(self, reward, probs):
        R = reward
        mask = R < 0
        probs = probs
        lpR1 = (1 - probs).log() * abs(R)
        lpR2 = probs.log() * abs(R)
        loss = -(lpR1[mask].sum() + lpR2[~mask].sum())
        return loss / probs.shape[0]

    def _sample_regret_actions(self, actions, nactions):
        new_actions = []
        for i in range(actions.shape[0]):
            cur_choices = [a for a in range(nactions)]
            cur_action = torch.tensor(cur_choices)
            new_actions.append(cur_action)

        new_actions = torch.stack(new_actions).to(actions.device)
        return new_actions

    def sample(self, attn_dist, training=True):
        dist = Categorical(attn_dist)
        if not training:
            action = torch.max(attn_dist, dim=1)[1].to(attn_dist.device)
            probs = dist.log_prob(action).exp()
            return probs, action

        random_actions = (torch.rand(attn_dist.shape[0]) * attn_dist.shape[-1]).long().to(attn_dist.device)
        rand_mask = (torch.rand(attn_dist.shape[0]) > self.policy_weight)
        action = dist.sample()
        action[rand_mask] = random_actions[rand_mask]

        probs = dist.log_prob(action).exp()
        self.policy_weight = self.policy_weight + (1 - self.policy_weight) * self.policy_epsilon

        return probs, action

    def sample_next_z(self, attn_dist, z, training=True):
        probs, action = self.sample(attn_dist, training)
        z2 = self.apply_action(z, action)
        self.reinforce_params.append(ReinforceParams(probs, z, z2, action, attn_dist))
        return z2

    def entropy_loss(self, attn):
        mean_dist = Categorical(attn.mean(0))
        dist = Categorical(attn)

        mean_entropy = mean_dist.entropy()
        dist_entropy = dist.entropy()
        return (dist_entropy / mean_entropy).mean()

    def combine_rewards(self, x2, real):

        for i, params in enumerate(self.reinforce_params):
            if params.reward is None:
                probs, dist, action = params.probs, params.cat_dist, params.actions

                target_params = {'x2': x2, 'z2': real}
                rewards = self.reward(params.z, action, dist, target_params)
                pure_reward = rewards

                params.reward = rewards
                params.pure_reward = pure_reward
                params.per_action_exp_reward = torch.tensor([0]).to(real.device)

    def combine_returns(self):
        for i, params in enumerate(self.reinforce_params):
            params.returns = params.reward * (self.reinforce_discount ** i)
            params.returns = params.returns + self.reinforce_params[i - 1].returns if i > 0 else params.returns

    def reduce_returns(self, reduction=0.25):
        for i, params in enumerate(self.reinforce_params):
            params.returns = params.returns - i * reduction

    def combine_reinforce_losses(self):
        for i, params in enumerate(self.reinforce_params):
            regret = params.regret * 0 if not self.use_regret else params.regret
            params.policy_loss = self.reinforce(params.reward - regret.detach(), params.probs)
            params.returns_loss = self.reinforce(params.returns - params.regret.detach(), params.probs)

    def regret(self, old_z, actions, dist, current_reward, target_params):
        new_actions = self._sample_regret_actions(actions, dist.probs.shape[-1])

        new_rewards = []
        for i in range(new_actions.shape[-1]):
            new_rewards.append(self.reward(old_z, new_actions[:, i], dist, target_params))

        new_rewards = torch.stack(new_rewards, -1)
        optimal_rewards, optimal_rewards_argmax = new_rewards.max(-1)
        per_action_reward = new_rewards.mean(0)
        expected_reward = new_rewards.mean(1)

        regret = optimal_rewards - current_reward

        return regret.mean(), per_action_reward, optimal_rewards_argmax, expected_reward, new_rewards

    def combine_regret(self, target_params):
        for i, params in enumerate(self.reinforce_params):
            if self.use_regret and params.regret is None:
                params.regret, params.per_action_reward, params.optimal_rewards, _, params.batch_per_action_rewards = self.regret(
                    params.z, params.actions, params.cat_dist, params.reward, target_params)
            else:
                params.regret = torch.tensor([0]).to(params.z.device)

    def apply_action(self, z, action):
        raise NotImplementedError

    def reward(self, old_z, action, dist, target_params):
        raise NotImplementedError


class VAEReinforceBase(ReinforceBase):
    def __init__(self, base_policy_weight, base_policy_epsilon, normalised_reward, use_regret,
                 decoder, rep_fn, multi_action_strategy='reward', reinforce_discount=0.99, entropy_weight=0.):
        super().__init__(base_policy_weight=base_policy_weight, base_policy_epsilon=base_policy_epsilon,
                         normalised_reward=normalised_reward, reinforce_discount=reinforce_discount,
                         use_regret=use_regret)
        self.decoder = decoder
        self.rep_fn = rep_fn
        self.multi_action_strategy = multi_action_strategy
        self.entropy_weight = entropy_weight

    def apply_action(self, z, action):
        return self.rep_fn(z, action)

    def _latent_reward(self, zs, z2s, real):
        mse_pre_action = (zs - real).pow(2).sum(-1)
        mse_post_action = (z2s - real).pow(2).sum(-1)
        return (mse_pre_action - mse_post_action).float().detach()

    def reward(self, old_z, action, dist, target_params):
        new_z = self.apply_action(old_z, action)
        true_z2 = target_params['z2']
        reward = self._latent_reward(old_z, new_z, true_z2)
        return reward.detach()

    def combine_entropy_loss(self):
        for i, params in enumerate(self.reinforce_params):
            params.entropy_ratio = self.entropy_loss(params.attn_dist)

    def entropy_explore_loss(self, weight=1.):
        loss = []
        for param in self.reinforce_params:
            loss.append(-param.cat_dist.entropy().mean() * weight)
        return sum(loss)

    def combiners(self, real, x2):
        self.combine_rewards(x2, real)
        self.combine_returns()
        self.reduce_returns(0.)
        self.combine_entropy_loss()
        self.combine_regret({'x2': x2, 'z2': real})
        self.combine_reinforce_losses()

    def loss(self, real, x2):
        self.combiners(real, x2)

        if self.multi_action_strategy == 'reward':
            reinforce_loss = sum([p.policy_loss for p in self.reinforce_params])
        else:
            reinforce_loss = sum([p.returns_loss for p in self.reinforce_params])

        entropy_explore = self.entropy_explore_loss(self.entropy_weight)

        out = {
            'reward_count': self.reinforce_params[-1].reward.mean(),
            'action_examples': self.reinforce_params[-1].actions,
            'rewards': self.reinforce_params[-1].reward
        }
        entropy_ratio = sum([p.entropy_ratio for p in self.reinforce_params])
        try:
            out['reinforce/paer'] = self.reinforce_params[-1].per_action_exp_reward
            out['reinforce/per_action_reward'] = sum([p.per_action_reward for p in self.reinforce_params])
            out['reinforce/optimal_rewards'] = sum([p.optimal_rewards for p in self.reinforce_params])
            out['reinforce/batch_per_action_rewards'] = self.reinforce_params[-1].batch_per_action_rewards
        except Exception as _:
            pass

        tb_log = {'reinforce/loss': reinforce_loss,
                  'reinforce/returns': sum([p.returns.mean() for p in self.reinforce_params]),
                  'reinforce/entropy_explore': entropy_explore,
                  'reinforce/policy_weight': torch.tensor(self.policy_weight),
                  'reinforce/entropy_loss': entropy_ratio}

        return reinforce_loss + entropy_explore, tb_log, out

    def regret_loss(self):
        regret = sum([p.regret for p in self.reinforce_params])
        return regret, {'reinforce/regret': regret}

    def reset(self):
        self.reinforce_params = []
