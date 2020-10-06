import argparse
import ast
from trainer import run

parser = argparse.ArgumentParser('Disentanglement')
# Basic Training Args
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='forward', type=str, choices=['beta_shapes', 'beta_celeb', 'forward', 'rgrvae', 'dip_vae_i', 'dip_vae_ii', 'beta_forward', 'dforward'])
parser.add_argument('--dataset', default='forward', type=str, choices=['flatland', 'dsprites'])
parser.add_argument('--data-path', default=None, type=str, help='Path to dataset root')
parser.add_argument('--latents', default=4, type=int, help='Number of latents')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--split', default=0.1, type=float, help='Validation split fraction')
parser.add_argument('--shuffle', default=True, type=ast.literal_eval, help='Shuffle dataset')
parser.add_argument('--lr-scheduler', default='none', choices=['exp', 'none'], type=str)
parser.add_argument('--lr-scheduler-gamma', default=0.99, type=float, help='Exponential lr scheduler gamma')

# Model Loading
parser.add_argument('--base-model', default='beta_forward', type=str, help='Base model for rgrvae, dforward')
parser.add_argument('--base-model-path', default=None, help='Path to base model state which is to be loaded')
parser.add_argument('--load-model', default=False, type=ast.literal_eval, help='Continue training by loading model')
parser.add_argument('--log-path', default=None, type=str, help='Path from which to load model')
parser.add_argument('--global-step', default=None, help='Set the initial logging step value', type=int)

# Learning Rates
parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--group-learning-rate', default=0.1, type=float, help='Learning rate for internal rgrvae matrix representations')
parser.add_argument('--policy-learning-rate', default=None, type=float, help='Learning rate for policy network')

# Hyperparams
parser.add_argument('--beta', default=1., type=float, help='Beta vae beta')
parser.add_argument('--capacity', default=None, type=float, help='KL Capacity')
parser.add_argument('--capacity-leadin', default=100000, type=int, help='KL capacity leadin')
parser.add_argument('--gvae-gamma', default=None, type=float, help='Weighting for prediction loss. =beta if None')

# Metrics And Vis
parser.add_argument('--visualise', default=True, type=ast.literal_eval, help='Do visualisations')
parser.add_argument('--metrics', default=False, type=ast.literal_eval, help='Calculate disentanglement metrics at each step')
parser.add_argument('--end-metrics', default=True, type=ast.literal_eval, help='Calculate disentanglement metrics at end of run')
parser.add_argument('--evaluate', default=False, type=ast.literal_eval, help='Only run evalulation')

# RGrVAE Args
parser.add_argument('--offset', default=1, type=int, help='Generative factor offset for each action')
parser.add_argument('--use-regret', default=False, type=ast.literal_eval, help='Use regret on reinforce models')
parser.add_argument('--base-policy-weight', default=1., type=float, help='Base weight at which to apply policy over random choice')
parser.add_argument('--base-policy-epsilon', default=0.0000, type=float)
parser.add_argument('--group-structure', default=['c+', 'c-'], type=str, nargs='+', help='Group structure per latent pair in group vae. Options in models/actions: GroupWiseAction')
parser.add_argument('--num-action-steps', default=1, type=int, help='Number of action steps to allow Reinforced GroupVAE')
parser.add_argument('--multi-action-strategy', default='reward', choices=['reward', 'returns'], help='Strategy for reinforcing multiple actions.')
parser.add_argument('--reinforce-discount', default=0.99, type=float, help='Discount factor for reinforce rewards/returns')
parser.add_argument('--use-cob', default=False, type=ast.literal_eval, help='Use change of basis for representaitons in gvae')
parser.add_argument('--entropy-weight', default=0.01, type=float, help='Entropy weight for RL exploration')
parser.add_argument('--noise-name', default=None, choices=['BG', 'Salt', 'Gaussian'])
args = parser.parse_args()


if __name__ == '__main__':
    from models.utils import ContextTimer
    with ContextTimer('Run'):
        run(args)
