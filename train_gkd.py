import matplotlib.pyplot as plt

import torch
import torch.optim as optim

from utils import accuracy, encode_onehot_torch, class_f1, roc_auc, half_normalize, loss, calculate_imbalance_weight
from models import Fully


def run_gkd(adj, features, labels, idx_train, idx_val, idx_test, idx_connected,
                 params_teacher, params_student, params_lpa, isolated_test, sample_weight, use_cuda=True, show_stats=False):

    def train_teacher(epoch):
        model_teacher.train()
        optimizer_teacher.zero_grad()
        output = model_teacher(features)
        loss_train = loss(output[idx_train], labels_one_hot[idx_train], sample_weight[idx_train])
        acc_train = accuracy(output[idx_train], labels[idx_train])
        stats['f1macro_train'].append(class_f1(output[idx_train], labels[idx_train], type='macro'))
        if 'auc_train' in stats:
            stats['auc_train'].append(roc_auc(output[idx_train, 1], labels[idx_train]))
        loss_train.backward()
        optimizer_teacher.step()

        model_teacher.eval()
        output = model_teacher(features)

        loss_val = loss(output[idx_val], labels_one_hot[idx_val], sample_weight[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if 'auc_val' in stats:
            stats['auc_val'].append(roc_auc(output[idx_val, 1], labels[idx_val]))
        stats['f1macro_val'].append(class_f1(output[idx_val], labels[idx_val], type='macro'))
        stats['loss_train'].append(loss_train.item())
        stats['acc_train'].append(acc_train.item())
        stats['loss_val'].append(loss_val.item())
        stats['acc_val'].append(acc_val.item())

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

        return output.detach()

    def test_teacher():
        model_teacher.eval()
        output = model_teacher(features)
        loss_test = loss(output[idx_test], labels_one_hot[idx_test], sample_weight[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        stats['f1macro_test'].append(class_f1(output[idx_test], labels[idx_test], type='macro'))
        stats['loss_test'].append(loss_test.item())
        stats['acc_test'].append(acc_test.item())
        if 'auc_test' in stats:
            stats['auc_test'].append(roc_auc(output[idx_test, 1], labels[idx_test]))

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))

    def train_student(epoch):
        model_student.train()
        optimizer_student.zero_grad()
        output = model_student(features)
        loss_train = loss(output[idx_lpa], labels_lpa[idx_lpa], sample_weight[idx_lpa])
        acc_train = accuracy(output[idx_lpa], labels[idx_lpa])
        stats['f1macro_train'].append(class_f1(output[idx_train], labels[idx_train], type='macro'))
        if 'auc_train' in stats:
            stats['auc_train'].append(roc_auc(output[idx_train, 1], labels[idx_train]))
        if 'f1binary_train' in stats:
            stats['f1binary_train'].append(class_f1(output[idx_train], labels[idx_train], type='binary'))
        loss_train.backward()
        optimizer_student.step()

        model_student.eval()
        output = model_student(features)
        loss_val = loss(output[idx_val], labels_one_hot[idx_val], sample_weight[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])
        if 'auc_val' in stats:
            stats['auc_val'].append(roc_auc(output[idx_val, 1], labels[idx_val]))
        stats['f1macro_val'].append(class_f1(output[idx_val], labels[idx_val], type='macro'))
        if 'f1binary_val' in stats:
            stats['f1binary_val'].append(class_f1(output[idx_val], labels[idx_val], type='binary'))
        stats['loss_train'].append(loss_train.item())
        stats['acc_train'].append(acc_train.item())
        stats['loss_val'].append(loss_val.item())
        stats['acc_val'].append(acc_val.item())

        print('Epoch: {:04d}'.format(epoch + 1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'acc_val: {:.4f}'.format(acc_val.item()))

    def test_student():
        model_student.eval()
        output = model_student(features)
        loss_test = loss(output[idx_test], labels_one_hot[idx_test], sample_weight[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
        stats['f1macro_test'].append(class_f1(output[idx_test], labels[idx_test], type='macro'))
        if 'f1binary_test' in stats:
            stats['f1binary_test'].append(class_f1(output[idx_test], labels[idx_test], type='binary'))
        stats['loss_test'].append(loss_test.item())
        stats['acc_test'].append(acc_test.item())
        if 'auc_test' in stats:
            stats['auc_test'].append(roc_auc(output[idx_test, 1], labels[idx_test]))

        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return

    stats = dict()
    states = ['train', 'val', 'test']
    metrics = ['loss', 'acc', 'f1macro','auc']

    for s in states:
        for m in metrics:
            stats[m + '_' + s] = []

    labels_one_hot = encode_onehot_torch(labels)
    idx_lpa = list(idx_connected) + list(idx_train)
    if isolated_test:
        idx_lpa = [x for x in idx_lpa if (x not in idx_test) and (x not in idx_val)]
    idx_lpa = torch.LongTensor(idx_lpa)
    model_teacher = Fully(nfeat=features.shape[1],
                nhid=params_teacher['hidden'],
                nclass=int(labels_one_hot.shape[1]),
                dropout=params_teacher['dropout'])
    model_student = Fully(nfeat=features.shape[1],
                nhid=params_student['hidden'],
                nclass=int(labels_one_hot.shape[1]),
                dropout=params_student['dropout'])

    optimizer_teacher = optim.Adam(model_teacher.parameters(),
                           lr=params_teacher['lr'],
                           weight_decay=params_teacher['weight_decay'])
    optimizer_student = optim.Adam(model_student.parameters(),
                           lr=params_student['lr'],
                           weight_decay=params_student['weight_decay'])

    if use_cuda:
        model_student.cuda()
        model_teacher.cuda()
        features = features.cuda()
        sample_weight = sample_weight.cuda()
        adj = adj.cuda()
        labels = labels.cuda()
        labels_one_hot = labels_one_hot.cuda()
        idx_train = idx_train.cuda()
        idx_val = idx_val.cuda()
        idx_test = idx_test.cuda()
        idx_lpa = idx_lpa.cuda()

    labels_lpa = labels_one_hot
    best_metric_val = 0
    for epoch in range(params_teacher['epochs']):
        output_fc = train_teacher(epoch)
        test_teacher()
        if stats[params_teacher['best_metric']][-1] >= best_metric_val and epoch > params_teacher['burn_out']:
            best_metric_val = stats[params_teacher['best_metric']][-1]
            best_output = output_fc

    alpha = params_lpa['alpha']
    labels_start = torch.exp(best_output)
    labels_lpa = labels_start
    labels_lpa[idx_train, :] = labels_one_hot[idx_train, :].float()
    for i in range(params_lpa['epochs']):
        labels_lpa = (1 - alpha) * adj.mm(labels_lpa) + alpha * labels_start
        labels_lpa[idx_train, :] = labels_one_hot[idx_train, :].float()
        labels_lpa = half_normalize(labels_lpa)

    sample_weight = calculate_imbalance_weight(idx_lpa, labels_lpa.argmax(1))
    if use_cuda:
        sample_weight = sample_weight.cuda()
    ### empty stats
    for k, v in stats.items():
        stats[k] = []

    for epoch in range(params_student['epochs']):
        train_student(epoch)
        test_student()

    if show_stats:
        plt.figure()
        fig, axs = plt.subplots(len(states), len(metrics), figsize=(5 * len(metrics), 3 * len(states)))
        for i, s in enumerate(states):
            for j, m in enumerate(metrics):
                axs[i, j].plot(stats[m + '_' + s])
                axs[i, j].set_title(m + ' ' + s)
        plt.show()

    return stats
