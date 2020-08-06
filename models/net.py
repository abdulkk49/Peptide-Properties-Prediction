import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidueNet(nn.Module):
    def ___init__(self, inplanes = 1024, outplanes = 32, stride = 1):
        super(ResidueNet, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv1d(inplanes, outplanes, kernel_size = 7, padding = 3, stride = self.stride, bias=False)
        self.relu = nn.relu()
        self.dropout = nn.Dropout(p = 0.25)
        # self.softmax = nn.Softmax(dim = -1)
        self.convQ3 = nn.Conv1d(outplanes, 8, kernel_size = 7, padding = 3, stride = self.stride, bias=False)
        self.convQ8 = nn.Conv1d(outplanes, 8, kernel_size = 7, padding = 3, stride = self.stride, bias=False)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        q3 = self.convQ3(x)
        q8 = self.convQ8(x)

        return F.log_softmax(q3,-2), F.log_softmax(q8,-2)
    
def loss_fn(log_probs, targ1hot):
    loss = torch.sum(-1 * targ1hot * log_probs)
    return loss

def accuracy(outputs, labels, mask):
    outputs = np.argmax(outputs, axis=-1)
    outputs = outputs.flatten()
    labels = np.argmax(labels, axis=-1)
    labels = labels.flatten()
    mask = mask.flatten().astype(np.bool)
    return np.sum(outputs[mask]==labels[mask])/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'q3accuracy': accuracy, 'q8accuracy': accuracy, 
    # could add more metrics such as accuracy for each token type
}

