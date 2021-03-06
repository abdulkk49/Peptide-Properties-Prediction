import random, torch
import os, numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


import random, torch
import os, numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


class PerResidueDataset(Dataset):
    def __init__(self, labelprefix, input_ids, attention_mask):
        self.labelprefix = labelprefix
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = np.load(labelprefix)['label']
        self.num_sample = self.labels.shape[0]
    
    def __getitem__(self, index):
       
        at_m = self.attention_mask[index]
        in_id = self.input_ids[index]
        q8label = self.labels[index,:,1:]
        mask = self.labels[index,:,0]
       
        return in_id, at_m, torch.Tensor(q8label), torch.Tensor(mask)

    def __len__(self):
        return self.num_sample





def fetch_dataloader(action, labelprefix, input_ids, attention_mask, params, collate_fn):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        action: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters
    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    # transformer = transforms.Compose([transforms.ToTensor()])
    dataloaders = {}
    if action == 'train':
        dataset = PerResidueDataset(labelprefix, input_ids, attention_mask)
        # batch_size = 128
        validation_split = .05
        shuffle_dataset = True
        random_seed= 42

        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset :
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = DataLoader(dataset, batch_size=params.batch_size, collate_fn = collate_fn, 
                                sampler=train_sampler, num_workers=params.num_workers)
        validation_loader = DataLoader(dataset, batch_size=params.batch_size, collate_fn = collate_fn,
                                        sampler=valid_sampler, num_workers=params.num_workers)
        dataloaders['train'] = train_loader
        dataloaders['val'] = validation_loader
    else:
        dataset = PerResidueDataset(labelprefix, embedprefix, transformer)
        test_loader = DataLoader(dataset, batch_size=params.batch_size, 
                                sampler=train_sampler, num_workers=params.num_workers, collate_fn = collate_fn)
        dataloaders['test'] = test_loader

    return dataloaders