"""Evaluates the model"""

import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
import utils
import models.net as net
import models.data_loader as data_loader

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='data/64x64_SIGNS',
                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='experiments/base_model',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best', help="name of the file in --model_dir \
                     containing weights to load")


def evaluate(model, loss_fn, dataloader, metrics, params):
    """Evaluate the model on `num_steps` batches.

    Args:
        model: (torch.nn.Module) the neural network
        loss_fn: a function that takes batch_output and batch_labels and computes the loss for the batch
        dataloader: (DataLoader) a torch.utils.data.DataLoader object that fetches data
        metrics: (dict) a dictionary of functions that compute a metric using the output and labels of each batch
        params: (Params) hyperparameters
        num_steps: (int) number of batches to train on, each of size params.batch_size
    """

    # set model to evaluation mode
    model.eval()

    # summary for current eval loop
    summ = []
    for i, (val_batch, q8labels_batch, mask) in enumerate(dataloader):
            # move to GPU if available
            if params.cuda:
                val_batch, q8labels_batch = val_batch.cuda(non_blocking=True),\
                q8labels_batch.cuda(non_blocking=True), mask.cuda(non_blocking = True)
            # N x 1634 x 1024 -> N x 1024 x 1632
            val_batch = val_batch.permute(0,2,1)
            # N x 1632 x 3
            q3labels_batch = torch.zeros((val_batch.shape[0],val_batch.shape[2],3))
            if params.cuda:
                q3labels_batch = q3labels_batch.cuda(non_blocking=True)

            q3labels_batch[:,:,0] = torch.sum(q8labels_batch[:,:,0:3], axis = -1)
            q3labels_batch[:,:,1] = torch.sum(q8labels_batch[:,:,3:5], axis = -1)
            q3labels_batch[:,:,2] = torch.sum(q8labels_batch[:,:,5:8], axis = -1)

            # compute model output and loss
            # N x 3 x 1632, N x 6 x 1632
            q3output_batch, q8output_batch = model(val_batch)
            # N x 3 x 1632 -> N x 1632 x 3
            q3output_batch = q3output_batch.permute(0,2,1)
            # N x 6 x 1632 -> N x 1632 x 6
            q8output_batch = q8output_batch.permute(0,2,1)

            q3loss = loss_fn(q3output_batch, q3labels_batch)
            q8loss = loss_fn(q8output_batch, q8labels_batch)

            loss = q3loss + q8loss
            
            # extract data from torch Variable, move to cpu, convert to numpy arrays
            q3output_batch = q3output_batch.data.cpu().numpy()
            q8output_batch = q8output_batch.data.cpu().numpy()
            q3labels_batch = q3labels_batch.data.cpu().numpy()
            q8labels_batch = q8labels_batch.data.cpu().numpy()

            # mask shape = N x 1632
            # compute all metrics on this batch
            summary_batch = {'val_q3accuracy': metrics['q3accuracy'](q3output_batch, q3labels_batch, mask), 'val_q8accuracy': metrics['q8accuracy'](q8output_batch, q8labels_batch, mask)}
            summary_batch['val_loss'] = loss.item()
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
    logging.info("- Validation metrics: " + metrics_string)
    return metrics_mean


if __name__ == '__main__':
    """
        Evaluate the model on the test set.
    """
    # Load the parameters
    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # use GPU if available
    params.cuda = torch.cuda.is_available()

    # Set the random seed for reproducible experiments
    torch.manual_seed(230)
    if params.cuda:
        torch.cuda.manual_seed(230)

    # Get the logger
    utils.set_logger(os.path.join(args.model_dir, 'evaluate.log'))

    # Create the input data pipeline
    logging.info("Creating the dataset...")

    # fetch dataloaders
    dataloaders = data_loader.fetch_dataloader(['test'], args.data_dir, params)
    test_dl = dataloaders['test']

    logging.info("- done.")

    # Define the model
    model = net.Net(params).cuda() if params.cuda else net.Net(params)

    loss_fn = net.loss_fn
    metrics = net.metrics

    logging.info("Starting evaluation..")

    # Reload weights from the saved file
    utils.load_checkpoint(os.path.join(
        args.model_dir, args.restore_file + '.pth.tar'), model)

    # Evaluate
    test_metrics = evaluate(model, loss_fn, test_dl, metrics, params)
    save_path = os.path.join(
        args.model_dir, "metrics_test_{}.json".format(args.restore_file))
    utils.save_dict_to_json(test_metrics, save_path)
