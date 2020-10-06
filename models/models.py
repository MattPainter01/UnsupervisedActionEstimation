from models.beta import BetaCeleb, BetaShapes
from models.forward_vae import forward_vae, beta_forward
from models.group_vae import rl_group_vae, forward_grvae
from models.factor_vae import factor_vae
from models.dip_vae import dip_vae

models = {
    'beta_shapes': BetaShapes,
    'beta_celeb': BetaCeleb,
    'forward': forward_vae,
    'rgrvae': rl_group_vae(False),
    'factor_vae': factor_vae,
    'dip_vae_i': dip_vae,
    'dip_vae_ii': dip_vae,
    'beta_forward': beta_forward,
    'dforward': forward_grvae(False),
}
