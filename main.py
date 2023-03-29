#!/usr/bin/env python3
from argparse import ArgumentParser
import numpy as np
import torch

from SETMLP import SETMLP
from data import get_dataset, DATASET_CONFIGS
from train import train
from model import MLP
import utils
import os

parser = ArgumentParser('EWC PyTorch Implementation')
parser.add_argument('--model', type=str, default='MLP')
parser.add_argument('--hidden-size', type=int, default=400)
parser.add_argument('--hidden-layer-num', type=int, default=2)
parser.add_argument('--hidden-dropout-prob', type=float, default=.5)
parser.add_argument('--input-dropout-prob', type=float, default=.2)

parser.add_argument('--task-number', type=int, default=8)
parser.add_argument('--epochs-per-task', type=int, default=3)
parser.add_argument('--lamda', type=float, default=40)
parser.add_argument('--lr', type=float, default=1e-1)
parser.add_argument('--weight-decay', type=float, default=0)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--test-size', type=int, default=1024)
parser.add_argument('--fisher-estimation-sample-size', type=int, default=1024)
parser.add_argument('--random-seed', type=int, default=0)
parser.add_argument('--no-gpus', action='store_false', dest='cuda')
parser.add_argument('--eval-log-interval', type=int, default=250)
parser.add_argument('--loss-log-interval', type=int, default=250)
parser.add_argument('--consolidate', action='store_true')
parser.add_argument('--ascTopologyChangePeriod', type=int, default=200)
parser.add_argument('--zeta', type=float, default=0.1)
parser.add_argument('--affix', type=str, default='')


if __name__ == '__main__':
    args = parser.parse_args()

    # decide whether to use cuda or not.
    cuda = torch.cuda.is_available() and args.cuda
    print(f'cuda available: {torch.cuda.is_available()}')

    # generate permutations for the tasks.
    np.random.seed(args.random_seed)
    permutations = [
        np.random.permutation(DATASET_CONFIGS['mnist']['size']**2) for
        _ in range(args.task_number)
    ]

    # prepare mnist datasets.
    train_datasets = [
        get_dataset('mnist', permutation=p) for p in permutations
    ]
    test_datasets = [
        get_dataset('mnist', train=False, permutation=p) for p in permutations
    ]
    if args.model == 'SETMLP':
    # prepare the model.
        model = SETMLP(
            DATASET_CONFIGS['mnist']['size']**2,
            DATASET_CONFIGS['mnist']['classes'],
            hidden_size=args.hidden_size,
            hidden_layer_num=args.hidden_layer_num,
            hidden_dropout_prob=args.hidden_dropout_prob,
            input_dropout_prob=args.input_dropout_prob,
            lamb_func=args.lamda,
            ascTopologyChangePeriod=args.ascTopologyChangePeriod,
            zeta=args.zeta,
            affix=args.affix
        )
    elif args.model == 'MLP':
        model = MLP(
            DATASET_CONFIGS['mnist']['size'] ** 2,
            DATASET_CONFIGS['mnist']['classes'],
            hidden_size=args.hidden_size,
            hidden_layer_num=args.hidden_layer_num,
            hidden_dropout_prob=args.hidden_dropout_prob,
            input_dropout_prob=args.input_dropout_prob,
            lamb_func=args.lamda,
        )
    # initialize the parameters.
    utils.xavier_initialize(model)

    # prepare the cuda if needed.
    if cuda:
        model.cuda()

    # run the experiment.
    train(
        model, train_datasets, test_datasets,
        epochs_per_task=args.epochs_per_task,
        batch_size=args.batch_size,
        test_size=args.test_size,
        consolidate=args.consolidate,
        fisher_estimation_sample_size=args.fisher_estimation_sample_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        eval_log_interval=args.eval_log_interval,
        loss_log_interval=args.loss_log_interval,
        cuda=cuda
    )
