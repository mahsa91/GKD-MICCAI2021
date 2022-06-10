import torch.nn as nn
import torch.nn.functional as F


class Fully(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout,multi_label=False):
        super(Fully, self).__init__()
        self.multi_label = multi_label
        layers = []
        if len(nhid) == 0:
            layers.append(nn.Linear(nfeat, nclass))
        else:
            layers.append(nn.Linear(nfeat, nhid[0]))
            for i in range(len(nhid) - 1):
                layers.append(nn.Linear(nhid[i], nhid[i + 1]))
            if nclass > 1:
                layers.append(nn.Linear(nhid[-1], nclass))
        self.fc = nn.ModuleList(layers)

        self.dropout = dropout
        self.nclass = nclass

    def forward(self, x, adj=None):
        end_layer = len(self.fc) - 1 if self.nclass > 1 else len(self.fc)
        for i in range(end_layer):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.fc[i](x)
            x = F.relu(x)

        classifier = self.fc[-1](x)
        if self.multi_label:
            classifier = classifier
        else:
            classifier = F.log_softmax(classifier, dim=1)
        return classifier


