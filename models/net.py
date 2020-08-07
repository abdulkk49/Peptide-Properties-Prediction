import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidueNet(nn.Module):
    def ___init__(self, params):
        super(ResidueNet, self).__init__()
        self.inplanes = params.inplanes
        self.outplanes = params.outplanes
        self.kernel = params.kernel
        self.padding = params.padding
        self.stride = params.stride
        self.drop = params.drop
        self.conv1 = nn.Conv1d(self.inplanes, self.outplanes, kernel_size = self.kernel, padding = self.padding, stride = self.stride, bias=False)
        self.relu = nn.relu()
        self.dropout = nn.Dropout(p = self.drop)
        self.convQ3 = nn.Conv1d(self.outplanes, 3, kernel_size = self.kernel, padding = self.padding, stride = self.stride, bias=False)
        self.convQ8 = nn.Conv1d(self.outplanes, 8, kernel_size = self.kernel, padding = self.padding, stride = self.stride, bias=False)

    def forward(self,x):
        # N x 1024 x 1632 -> N x 32 x 1632
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # N x 32 x 1632 -> N x 3 x 1632
        q3 = self.convQ3(x)
        # N x 32 x 1632 -> N x 8 x 1632
        q8 = self.convQ8(x)

        return F.log_softmax(q3,-2), F.log_softmax(q8,-2)
    
def loss_fn(log_probs, targ1hot):
    # (N x 1632 x 8 or 3) * (N x 1632 x 8 or 3)
    loss = torch.sum(-1 * targ1hot * log_probs)
    return loss

def accuracy(outputs, labels, mask):
    # N x 1632 x 8(3) -> N x 1632
    outputs = np.argmax(outputs, axis=-1)
    # N x 1632 -> N*1632
    outputs = outputs.flatten()
    # N x 1632 x 8(3) -> N x 1632
    labels = np.argmax(labels, axis=-1)
    # N x 1632 -> N*1632
    labels = labels.flatten()
    # N x 1632 -> N*1632
    mask = mask.flatten().astype(np.bool)
    return np.sum(outputs[mask]==labels[mask])/float(np.sum(mask))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'q3accuracy': accuracy, 'q8accuracy': accuracy, 
    # could add more metrics such as accuracy for each token type
}

