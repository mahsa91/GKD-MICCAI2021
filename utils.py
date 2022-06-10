import torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def class_f1(output, labels, type='micro', pos_label=1):
    preds = output.max(1)[1].type_as(labels)
    return f1_score(labels.detach().cpu().numpy(), preds.cpu(), average=type, pos_label=pos_label)


def roc_auc(output, labels):
    return roc_auc_score(labels.cpu().numpy(), output.detach().cpu().numpy())


def loss(output,labels, weights=None):
    if weights is None:
        weights = torch.ones(labels.shape[0])
    return torch.sum(- weights * (labels.float() * output).sum(1), -1)


def half_normalize(mx):
    rowsum = mx.sum(1).float()
    r_inv = rowsum.pow(-1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diag(r_inv)
    mx = r_mat_inv.mm(mx)
    return mx


def encode_onehot_torch(labels,num_classes=None):
    if num_classes is None:
        num_classes = int(labels.max() + 1)
    y = torch.eye(num_classes)
    return y[labels]


def calculate_imbalance_weight(idx,labels):
    weights = torch.ones(len(labels))
    for i in range(labels.max()+1):
        sub_node = torch.where(labels == i)[0]
        sub_idx = [x.item() for x in sub_node if x in idx]
        weights[sub_idx] = 1 - len(sub_idx)/ len(idx)
    return weights





