import torch
torch.cuda.current_device()

import os # check?
import os.path as osp
import numpy as np
from tqdm import tqdm

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

from knockoffnets.knockoff import datasets
from knockoffnets.knockoff.victim.blackbox import Blackbox
from knockoffnets.knockoff.adversary.transfer import RandomAdversary

import knockoffnets.knockoff.config as cfg
import knockoffnets.knockoff.utils.model as model_utils
import knockoffnets.knockoff.utils.utils as knockoff_utils
import knockoffnets.knockoff.utils.transforms as transform_utils

torch.manual_seed(cfg.DEFAULT_SEED)

def random_sampler(blackbox_dir, n_images, batch_size = 64, queryset = "TinyImageNet200"):
    """
    Inputs:
        queryset - name of dataset to sample from (str)
        blackbox_dir - location where victim model is saved with checkpoint.pth.tar and params.jason
        batch_size - batch size for fetching images in parallel
        n_images - number of images to randomly query

    """
    # ----------- Set up queryset
    valid_datasets = datasets.__dict__.keys()
    if queryset not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    modelfamily = datasets.dataset_to_modelfamily[queryset_name] if params['modelfamily'] is None else params['modelfamily']
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    if queryset_name == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset_name](root=params['root'], transform=transform)
    else:
        queryset = datasets.__dict__[queryset_name](train=True, transform=transform)

    # ----------- Initialize blackbox
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    adversary = RandomAdversary(blackbox, queryset, batch_size = batch_size)

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(n_images)

    return transferset

