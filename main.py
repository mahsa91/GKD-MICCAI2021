import torch
import numpy as np
import os
import random

import argparse
import pickle
from train_gkd import run_gkd
from utils import calculate_imbalance_weight


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def last_argmax(l):
    reverse = l[::-1]
    return len(reverse) - np.argmax(reverse) - 1


def find_isolated(adj, idx_train):
    adj = adj.to_dense()
    deg = adj.sum(1)
    idx_connected = torch.where(deg > 0)[0]
    idx_connected = [x for x in idx_connected if x not in idx_train]
    return idx_connected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=42, type=int, help='Seed')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA if available.')
    parser.add_argument('--epochs_teacher', default=300, type=int, help='Number of epochs to train teacher network')
    parser.add_argument('--epochs_student', default=200, type=int, help='Number of epochs to train student network')
    parser.add_argument('--epochs_lpa', default=10, type=int, help='Number of epochs to train student network')

    parser.add_argument('--lr_teacher', type=float, default=0.005, help='Learning rate for teacher network.')
    parser.add_argument('--lr_student', type=float, default=0.005, help='Learning rate for student network.')

    parser.add_argument('--wd_teacher', type=float, default=5e-4, help='Weight decay for teacher network.')
    parser.add_argument('--wd_student', type=float, default=5e-4, help='Weight decay for student network.')

    parser.add_argument('--dropout_teacher', type=float, default=0.3, help='Dropout for teacher network.')
    parser.add_argument('--dropout_student', type=float, default=0.3, help='Dropout for student network.')

    parser.add_argument('--burn_out_teacher', default=100, type=int, help='Number of epochs to drop for selecting best \
                                                                    parameters based on validation set for teacher network')
    parser.add_argument('--burn_out_student', default=100, type=int, help='Number of epochs to drop for selecting best \
                                                                    parameters based on validation set for student network')

    parser.add_argument('--alpha', default=0.1, type=float, help='alpha')

    args = parser.parse_args()
    seed = args.seed
    use_cuda = args.use_cuda and torch.cuda.is_available()
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Torch RNG
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Python RNG
    np.random.seed(seed)
    random.seed(seed)
    ### Following lists shows the number of hidden neurons in each hidden layer of teacher network and student network respectively
    hidden_teacher = [8]
    hidden_student = [4]
    ### select the best output of teacher for lpa and best output of student for reporting based on the following metrics
    best_metric_teacher = 'f1macro_val' ### concat a member of these lists: [loss, acc, f1macro][train, val, test]
    best_metric_student = 'f1macro_val'

    ### show the changes of statistics in training, validation and test set with figures
    show_stats = False
    ### This is needed is you want to make sure that access to test set is not available during the training
    isolated_test = True

    ### Data should be loaded here as follow:
    ### adj is a sparse tensor showing the adjacency matrix between nodes with size N by N
    ### Features is a tensor with size N by F
    ### labels is a list of node labels
    ### idx train is a list contains the index of training samples. idx_val and idx_test follow the same pattern
    with open('./data/synthetic/sample-2000-f_128-g_4-gt_0.5-sp_0.5-100-200-500.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        adj, features, labels, idx_train, idx_val, idx_test = pickle.load(f)
    ### you can load weights for samples in the training set or calculate them here.
    ### the weights of nodes will be calculates in the train_gkd after the lpa step
    sample_weight = calculate_imbalance_weight(idx_train, labels)

    idx_connected = find_isolated(adj, idx_train)

    params_teacher = {
        'lr': args.lr_teacher,
        'hidden': hidden_teacher,
        'weight_decay': args.wd_teacher,
        'dropout': args.dropout_teacher,
        'epochs': args.epochs_teacher,
        'best_metric': best_metric_teacher,
        'burn_out': args.burn_out_teacher
    }
    params_student = {
        'lr': args.lr_student,
        'hidden': hidden_student,
        'weight_decay': args.wd_student,
        'dropout': args.dropout_teacher,
        'epochs': args.epochs_student,
        'best_metric': best_metric_student,
        'burn_out': args.burn_out_student
    }
    params_lpa = {
        'epochs': args.epochs_lpa,
        'alpha': args.alpha
    }

    stats = run_gkd(adj, features, labels, idx_train,
                        idx_val,idx_test, idx_connected, params_teacher, params_student, params_lpa,
                        sample_weight=sample_weight, isolated_test=isolated_test,
                        use_cuda=use_cuda, show_stats=show_stats)

    ind_val_max = params_student['burn_out'] + last_argmax(stats[params_student['best_metric']][params_student['burn_out']:])
    print('Last index with maximum metric on validation set: ' + str(ind_val_max))
    print('Final accuracy on val: ' + str(stats['acc_test'][ind_val_max]))
    print('Final Macro F1 on val: ' + str(stats['f1macro_test'][ind_val_max]))
    print('Final AUC on val' + str(stats['auc_test'][ind_val_max]))
