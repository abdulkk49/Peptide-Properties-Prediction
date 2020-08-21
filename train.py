"""Train the model"""

import argparse
import logging
import os, sys

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

import utils
from evaluate import evaluate
from os.path import join, exists, dirname, abspath, realpath

sys.path.append(dirname(abspath("__file__")))

from models.data_loader import *
from models.net import *
from transformers import BertModel, BertTokenizer
import re
import os
import requests, h5py
from tqdm.auto import tqdm
from os.path import join, exists, dirname, abspath, realpath

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./Embeddings',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='./trial',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--batch', default=128,
                    help="Batch Size for Training")
parser.add_argument('--num_workers', default=1,
                    help="Num workers for Training")

def download_file(url, filename):
    response = requests.get(url, stream=True)
    with tqdm.wrapattr(open(filename, "wb"), "write", miniters=1,
                        total=int(response.headers.get('content-length', 0)),
                        desc=filename) as fout:
        for chunk in response.iter_content(chunk_size=4096):
            fout.write(chunk)


def embedModel():
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    #Pretrained Model files
    modelUrl = 'https://www.dropbox.com/s/dm3m1o0tsv9terq/pytorch_model.bin?dl=1'
    configUrl = 'https://www.dropbox.com/s/d3yw7v4tvi5f4sk/bert_config.json?dl=1'
    vocabUrl = 'https://www.dropbox.com/s/jvrleji50ql5m5i/vocab.txt?dl=1'

    #Setting folder paths
    downloadFolderPath = 'models/ProtBert/'
    modelFolderPath = downloadFolderPath

    #Setting file paths
    modelFilePath = os.path.join(modelFolderPath, 'pytorch_model.bin')
    configFilePath = os.path.join(modelFolderPath, 'config.json')
    vocabFilePath = os.path.join(modelFolderPath, 'vocab.txt')

    #Creading model directory
    if not os.path.exists(modelFolderPath):
        os.makedirs(modelFolderPath)

    #Downloading pretrained model
    if not os.path.exists(modelFilePath):
        download_file(modelUrl, modelFilePath)
    if not os.path.exists(configFilePath):
        download_file(configUrl, configFilePath)
    if not os.path.exists(vocabFilePath):
        download_file(vocabUrl, vocabFilePath)

    #Initializing Tokenizer, Model
    tokenizer = BertTokenizer(vocabFilePath, do_lower_case=False )
    model = BertModel.from_pretrained(modelFolderPath)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model, tokenizer



bertModel, tokenizer = embedModel()
bertModel = bertModel.eval()

def collate_fn(batch):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_ids = [item[0] for item in batch]
    attention_mask = [item[1] for item in batch]
    input_ids = torch.stack(input_ids)
    attention_mask = torch.stack(attention_mask)
    print(input_ids.shape, attention_mask.shape)
    embedding = torch.zeros((input_ids.shape[0],1632,1024), dtype = torch.float32).to(device)
    i = 0
    bs = 16
    x = 0
    with torch.no_grad():
        while i + bs <= len(input_ids):
            e = bertModel(input_ids=input_ids[i:i+bs],attention_mask=attention_mask[i:i+bs])[0]
            # print(e.shape)
            embedding[x:x+bs,:,:] = e[:,1:1633,:]
            x += bs
            i += bs
            print(e.shape, i)

    with torch.no_grad(): 
    	if i != len(input_ids):
            #Final batch < 16
            e = bertModel(input_ids=input_ids[i:len(input_ids)],attention_mask=attention_mask[i:len(input_ids)])[0]
            embedding[x:x+len(input_ids) - i,:,:] = e[:,1:1633,:]
            print(e.shape, i)
    	

    q8label = torch.stack([item[2] for item in batch])
    mask = torch.stack([item[3] for item in batch])
    print(q8label.shape, mask.shape)
    return embedding, q8label, mask


def train(model, optimizer, loss_fn, dataloader, metrics, params):
    """Train the model on `num_steps` batches
    Args:
        model: (torch.nn.Module) the neural network
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
    """

    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    summ = []
    loss_avg = utils.RunningAverage()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, q8labels_batch, mask) in enumerate(dataloader):
            # move to GPU if available
            # if params.cuda:
            #     train_batch, q8labels_batch, mask = train_batch.cuda(non_blocking=True),\
            #     q8labels_batch.cuda(non_blocking=True), mask.cuda(non_blocking = True)
            # N x 1634 x 1024 -> N x 1024 x 1632
            # print(train_batch.shape, q8labels_batch.shape, mask.shape)
            train_batch = train_batch.permute(0,2,1)
            # N x 1632 x 3
            q3labels_batch = torch.zeros((train_batch.shape[0],train_batch.shape[2],3))
            # if params.cuda:
            #     q3labels_batch = q3labels_batch.cuda(non_blocking=True)
            print(q3labels_batch.size, q3labels_batch.device, train_batch.shape)
            q3labels_batch[:,:,0] = torch.sum(q8labels_batch[:,:,0:3], axis = -1)
            q3labels_batch[:,:,1] = torch.sum(q8labels_batch[:,:,3:5], axis = -1)
            q3labels_batch[:,:,2] = torch.sum(q8labels_batch[:,:,5:8], axis = -1)

            # compute model output and loss
            # N x 3 x 1632, N x 8 x 1632
            q3output_batch, q8output_batch = model(train_batch)

            # N x 3 x 1632 -> N x 1632 x 3
            q3output_batch = q3output_batch.permute(0,2,1)
            # N x 8 x 1632 -> N x 1632 x 8
            q8output_batch = q8output_batch.permute(0,2,1)

            q3loss = loss_fn(q3output_batch.cpu(), q3labels_batch)
            q8loss = loss_fn(q8output_batch.cpu(), q8labels_batch)

            loss = q3loss + q8loss
            # clear previous gradients, compute gradients of all variables wrt loss
            
            optimizer.zero_grad()
            loss.backward()

            # performs updates using calculated gradients
            optimizer.step()

            # Evaluate summaries only once in a while
            if i % params.save_summary_steps == 0:
                # extract data from torch Variable, move to cpu, convert to numpy arrays
                q3output_batch = q3output_batch.data.cpu().numpy()
                q8output_batch = q8output_batch.data.cpu().numpy()
                q3labels_batch = q3labels_batch.data.numpy()
                q8labels_batch = q8labels_batch.data.numpy()
                mask = mask.numpy()
                #mask shape = N x 1632
                # compute all metrics on this batch
                summary_batch = {'q3accuracy': metrics['q3accuracy'](q3output_batch, q3labels_batch, mask), 'q8accuracy': metrics['q8accuracy'](q8output_batch, q8labels_batch, mask)}
                summary_batch['loss'] = loss.item()
                summ.append(summary_batch)

            # update the average loss
            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

    # compute mean of all metrics in summary
    metrics_mean = {metric: np.mean([x[metric]
                                     for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v)
                                for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)


def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, metrics, params, model_dir,
                       restore_file=None):
    """Train the model and evaluate every epoch.
    Args:
        model: (torch.nn.Module) the neural network
        train_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches training data
        val_dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches validation data
        optimizer: (torch.optim) optimizer for parameters of model
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        model_dir: (string) directory containing config, weights and log
        restore_file: (string) optional- name of file to restore from (without its extension .pth.tar)
    """
    # reload weights from restore_file if specified
    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        logging.info("Restoring parameters from {}".format(restore_path))
        utils.load_checkpoint(restore_path, model, optimizer)

    best_val_q8acc = 0.0
    best_val_q3acc = 0.0

    for epoch in range(params.num_epochs):
        # Run one epoch
        logging.info("Epoch {}/{}".format(epoch + 1, params.num_epochs))

        # compute number of batches in one epoch (one full pass over the training set)
        train(model, optimizer, loss_fn, train_dataloader, metrics, params)

        # Evaluate for one epoch on validation set
        val_metrics = evaluate(model, loss_fn, val_dataloader, metrics, params)

        val_q8acc = val_metrics['val_q8accuracy']
        val_q3acc = val_metrics['val_q3accuracy']
        is_q8best = val_q8acc >= best_val_q8acc
        is_q3best = val_q3acc >= best_val_q3acc

        # Save weights
        utils.save_checkpoint({'epoch': epoch + 1,
                               'state_dict': model.state_dict(),
                               'optim_dict': optimizer.state_dict()},
                                is_q3best = is_q3best,
                                is_q8best = is_q8best,
                                checkpoint = model_dir)

        # If best_eval, best_save_path
        if is_q8best:
            logging.info("- Found new best q8 accuracy")
            best_val_q8acc = val_q8acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_q8best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)

        if is_q3best:
            logging.info("- Found new best q3 accuracy")
            best_val_q3acc = val_q3acc

            # Save best val metrics in a json file in the model directory
            best_json_path = os.path.join(
                model_dir, "metrics_val_q3best_weights.json")
            utils.save_dict_to_json(val_metrics, best_json_path)


        # Save latest val metrics in a json file in the model directory
        last_json_path = os.path.join(
            model_dir, "metrics_val_last_weights.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

if __name__ == '__main__':

    # Load the parameters from json file
    pwd = dirname(realpath("__file__"))
    print("Present Working Directory: ", pwd)
    args = parser.parse_args()
    json_path = os.path.join(pwd, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()
    params.batch_size = int(args.batch)
    params.num_workers = int(args.num_workers)
    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)
    print("Params: " ,params.__dict__)
    labelprefix = join(pwd,'maskandlabels.npz')
    # embeddir = join(pwd, 'Embeddings')
    # embedprefix = join(embeddir, 'batch')

    #Load sequences
    sequences_Example =[]
    count = 0
    with open("./residuesequences.txt", "r") as f:
        for seq in f.readlines():
            desc = str(seq).rstrip('\n')
            sequences_Example.append(desc)
            count += 1
    print("Total data points(Clean): ", str(count))

    #Replace "UZOB" with "X"
    sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #Tokenizing input sequences
    ids = tokenizer.batch_encode_plus(sequences_Example, add_special_tokens=True, pad_to_max_length=True)
    input_ids = torch.tensor(ids['input_ids']).to(device)
    attention_mask = torch.tensor(ids['attention_mask']).to(device)
    # Set the logger

    utils.set_logger(os.path.join(args.model_dir, 'train.log'))

    # Create the input data pipeline
    logging.info("Loading the datasets...")
    dataloaders = fetch_dataloader('train', labelprefix, input_ids, attention_mask, params, collate_fn)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    logging.info("- done.")

    # Define the model and optimizer
    model = ResidueNet(params)
    if params.cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    # fetch loss function and metrics
    loss_fn = loss_fn
    metrics = metrics

    # Train the model
    logging.info("Starting training for {} epoch(s)".format(params.num_epochs))
    train_and_evaluate(model, train_dl, val_dl, optimizer, loss_fn, metrics, params, join(pwd, "trial"),
                        None)
