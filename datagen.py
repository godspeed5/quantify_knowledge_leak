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
from knockoffnets.knockoff.adversary.train import samples_to_transferset

import knockoffnets.knockoff.config as cfg
import knockoffnets.knockoff.utils.model as model_utils
import knockoffnets.knockoff.utils.utils as knockoff_utils
import knockoffnets.knockoff.utils.transforms as transform_utils

torch.manual_seed(cfg.DEFAULT_SEED)

def random_sampler(blackbox_dir, n_images, img_folder = None, batch_size = 64, 
                   model_family = "imagenet", query_set = "TinyImageNet200"):
    """
    Inputs:
        blackbox_dir - location of victim model
        n_images - number of images to randomly query
        img_folder - location of img folder. REQD for custom query_set
        batch_size - batch size for fetching images in parallel
        model_family - choose from {imagenet, mnist, cifar}
        query_set - name of dataset to sample from (str) 
        
    Outputs:
        transferset - list of tuples (img loc, img tensor)
    """
    # ----------- Set up query_set
    valid_datasets = datasets.__dict__.keys()
    if query_set not in valid_datasets:
        raise ValueError('Dataset not found. Valid arguments = {}'.format(valid_datasets))
    
    modelfamily = datasets.dataset_to_modelfamily[query_set] if model_family is None else model_family
    transform = datasets.modelfamily_to_transforms[modelfamily]['test']
    
    if query_set == 'ImageFolder':
        assert params['root'] is not None, 'argument "--root ROOT" required for ImageFolder'
        query_set = datasets.__dict__[query_set](root = img_folder, transform = transform)
    else:
        query_set = datasets.__dict__[query_set](train = True, transform = transform)

    # ----------- Initialize blackbox
    blackbox = Blackbox.from_modeldir(blackbox_dir, device)

    # ----------- Initialize adversary
    adversary = RandomAdversary(blackbox, query_set, batch_size = batch_size)

    print('=> constructing transfer set...')
    transferset = adversary.get_transferset(n_images)

    return transferset


def get_training_set(blackbox_dir, n_images, original_dataset = "Caltech256", img_folder = None, 
                     batch_size = 64, model_family = "imagenet", query_set = "TinyImageNet200", label_only = True):
    """
    Inputs:
        blackbox_dir - location of victim model
        n_images - number of images to randomly query
        original_dataset - dataset the victim model was trained on
        img_folder - location of img folder. REQD for custom query_set
        batch_size - batch size for fetching images in parallel
        model_family - choose from {imagenet, mnist, cifar}
        query_set - name of dataset to sample from (str)
        label_only - use label only or probabilities of classes also? 
        
    Outputs:
        transferset - random sampled dataset (can be loaded into DataLoader directly)
    """
    # ----------- Set up transferset
    transferset_samples = random_sampler(blackbox_dir, n_images, img_folder, batch_size, model_family, query_set)
    
    # ----------- Clean up transfer (if necessary)
    if label_only:
        new_transferset_samples = []
        print('=> Using argmax labels (instead of posterior probabilities)')
        for i in range(len(transferset_samples)):
            x_i, y_i = transferset_samples[i]
            argmax_k = y_i.argmax()
            y_i_1hot = torch.zeros_like(y_i)
            y_i_1hot[argmax_k] = 1.
            new_transferset_samples.append((x_i, y_i_1hot))
        transferset_samples = new_transferset_samples

    # ----------- Set up testset
    modelfamily = datasets.dataset_to_modelfamily[original_dataset]
    transform = datasets.modelfamily_to_transforms[model_family]['test']
    transferset = samples_to_transferset(transferset_samples, budget = n_images, transform = transform)
    return transferset


if __name__ == '__main__':
    # ----------- HOW TO USE
    transferset = get_training_set("knockoffnets/models/victim/caltech256-resnet34", 10)

