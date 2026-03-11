# config/cub200_config_5way_1shot.py

from collections import OrderedDict

config = OrderedDict()
config['seed'] = 24
config['dataset_name'] = 'cub200'
dataset_config = OrderedDict()
dataset_config['root'] = '/home/chinchi/home/chinchi/code/dataset'
config['dataset'] = dataset_config
config['model_name'] = 'dgcm'
config['backbone'] = 'ViT-B/16'

dsp_config = OrderedDict()
dsp_config['dsp_iterations'] = 2
dsp_config['k_neighbors'] = 5
dsp_config['alpha_init'] = 0.95
dsp_config['beta_init'] = 0.5
dsp_config['tau_instance'] = 0.02
dsp_config['tau_proto'] = 0.02
dsp_config['tau_guidance'] = 0.02
config['dsp_config'] = dsp_config

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 1
train_opt['batch_size'] = 4
train_opt['iteration'] = 6000
train_opt['lr'] = 1e-4
train_opt['weight_decay'] = 5e-4
train_opt['dec_lr'] = 3000
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.5
train_opt['label_smoothing'] = 0.0
config['train_config'] = train_opt

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 1
eval_opt['batch_size'] = 4
eval_opt['iteration'] = 1000
eval_opt['interval'] = 1000
eval_opt['num_query'] = 15
config['eval_config'] = eval_opt