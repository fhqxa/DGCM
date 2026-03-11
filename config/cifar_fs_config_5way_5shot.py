# config/cifar_fs_config_5way_5shot.py

from collections import OrderedDict

config = OrderedDict()
config['seed'] = 24
config['dataset_name'] = 'cifar-fs'
dataset_config = OrderedDict()
dataset_config['root'] = '/home/chinchi/home/chinchi/code/dataset'
config['dataset'] = dataset_config
config['model_name'] = 'dgcm'
config['backbone'] = 'ViT-B/16'

dsp_config = OrderedDict()
dsp_config['dsp_iterations'] = 2
dsp_config['k_neighbors'] = 10
dsp_config['alpha_init'] = 0.5
dsp_config['beta_init'] = 0.5
dsp_config['tau_instance'] = 0.07
dsp_config['tau_proto'] = 0.07
dsp_config['tau_guidance'] = 0.07
config['dsp_config'] = dsp_config

train_opt = OrderedDict()
train_opt['num_ways'] = 5
train_opt['num_shots'] = 5
train_opt['batch_size'] = 10
train_opt['iteration'] = 10000
train_opt['lr'] = 1e-4
train_opt['weight_decay'] = 5e-5
train_opt['dec_lr'] = 6000
train_opt['lr_adj_base'] = 0.5
train_opt['dropout'] = 0.1
train_opt['label_smoothing'] = 0.0
config['train_config'] = train_opt

eval_opt = OrderedDict()
eval_opt['num_ways'] = 5
eval_opt['num_shots'] = 5
eval_opt['batch_size'] = 6
eval_opt['iteration'] = 1000
eval_opt['interval'] = 1000
eval_opt['num_query'] = 15
config['eval_config'] = eval_opt