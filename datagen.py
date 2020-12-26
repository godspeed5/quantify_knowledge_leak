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

def random_sampler(blackbox_dir, n_images, img_folder = None, batch_size = 64, model_family = None, queryset = "TinyImageNet200"):
    """
    Inputs:
        blackbox_dir - location of victim model
        n_images - number of images to randomly query
        img_folder - location of img folder. REQD for custom queryset
        batch_size - batch size for fetching images in parallel
        model_family - type of model architecture (eg. resnet34)
        queryset - name of dataset to sample from (str) 
        
    Outputs:
        transferset - random input tensors in a list
    """
    # ----------- Set up queryset
    valid_datasets = datasets.__dict__.keys()
    if queryset not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    
    modelfamily = datasets.dataset_to_modelfamily[queryset] if model_family is None else model_family
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    
    if queryset == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        queryset = datasets.__dict__[queryset](root=params['root'], transform = transform)
    else:
        queryset = datasets.__dict__[queryset](train=True, transform = transform)

    # ----------- Initialize blackbox
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    adversary = RandomAdversary(blackbox, queryset, batch_size = batch_size)

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(n_images)

    return transferset


if __name__ == '__main__':
    transferset = random_sampler("knockoffnets/models/victim/caltech256-resnet34", 10)

