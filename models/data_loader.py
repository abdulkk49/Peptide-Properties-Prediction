import random, torch
import os, numpy as np
import h5py
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import torchvision.transforms as transforms


class PerResidueDataset(Dataset):
    def __init__(self, labelprefix, embedprefix):
        self.labelprefix = labelprefix
        self.embedprefix = embedprefix
        self.labels = np.load(labelprefix)['label']
        self.num_sample = self.labels.shape[0]
    
    def __getitem__(self, index):
        bs = 128
        cnt = index // bs
        i = index % bs
        with h5py.File(self.embedprefix + str(cnt + 1) + ".h5",'r') as dataall:
            f  = dataall['embed']
            seq = f[index,:,:]
        q8label = self.labels[index,:,1:]
        mask = self.labels[index,:,0]
        seq = torch.Tensor(seq).float()
        q8label = torch.Tensor(q8label)
        mask = torch.Tensor(mask)
        return seq, q8label, mask

    def __len__(self):
        return self.num_sample





def fetch_dataloader(action, labelprefix, embedprefix, params):
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
        dataset = PerResidueDataset(labelprefix, embedprefix)
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

        train_loader = DataLoader(dataset, batch_size=params.batch_size, 
                                sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)
        validation_loader = DataLoader(dataset, batch_size=params.batch_size,
                                        sampler=valid_sampler, num_workers=params.num_workers, pin_memory=params.cuda)
        dataloaders['train'] = train_loader
        dataloaders['val'] = validation_loader
    else:
        dataset = PerResidueDataset(labelprefix, embedprefix, transformer)
        test_loader = DataLoader(dataset, batch_size=params.batch_size, 
                                sampler=train_sampler, num_workers=params.num_workers, pin_memory=params.cuda)
        dataloaders['test'] = test_loader

    return dataloaders