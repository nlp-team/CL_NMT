import os
import cifar10.model_loader
import torch

def load(dataset, model_name, model_file, data_parallel=False):
    if dataset == 'cifar10':
        net = cifar10.model_loader.load(model_name, model_file, data_parallel)
    elif dataset == 'minist':
        
        # Model
        input_dim = 784
        hidden_dim = 256
        output_dim = 10
        if model_name == 'logist':
            net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, output_dim)
            )
        else:
            # 'multiLayer'
            net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim,hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, output_dim)
                )

        if model_file:
            assert os.path.exists(model_file), model_file + " does not exist."
            stored = torch.load(model_file, map_location=lambda storage, loc: storage)
            if 'state_dict' in stored.keys():
                net.load_state_dict(stored['state_dict'])
            else:
                net.load_state_dict(stored)
    else:
        pass
    return net
