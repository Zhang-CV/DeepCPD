import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from data import Satellite
from model import DeepCPD
from tensorboardX import SummaryWriter


#########################################################
## Input Parameters
#########################################################
parser = argparse.ArgumentParser(description='PyTorch Implementation of DeepCPD')
## GPU
parser.add_argument('--cuda', type=int, default=1,
                    help='use of cuda (default: 1)')
## Training
parser.add_argument('--epochs', type=int, default=180,
                    help='number of total epochs to run (default: 180)')
parser.add_argument('--batch_size_train', default=16, type=int,
                    help='mini-batch size of training (default: 16)')
parser.add_argument('--batch_size_val', default=4, type=int,
                    help='mini-batch size of validation (default: 4)')
parser.add_argument('--batch_size_test', default=1, type=int,
                    help='mini-batch size of training (default: 16)')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--decay_epoch', default=-1, type=int, 
                    help='Reduces the learning rate every decay_epoch')
parser.add_argument('--lr_decay', default=0.5, type=float,
                    help='Learning rate decay for training (default: 0.5)')
parser.add_argument('--num_clusters', type=int, default=16,
                    help='number of classes (default: 16)')
parser.add_argument('--scalar', default=7.3951, type=float,
                    help='Normalization parameters for Space Shuttle Orbiter')
parser.add_argument('--eval', action='store_true', default=False, help='test the model')
parser.add_argument('--icp', action='store_true', default=False, help='Using ICP refinement')
parser.add_argument('--model_path', type=str, default='', metavar='N', help='Pretrained model path')


args = parser.parse_args()

#########################################################
## Read Data
#########################################################
if __name__ == '__main__':
    train_loader = DataLoader(Satellite(partition='train', scalar=args.scalar, gaussian_noise=False),
                              batch_size=args.batch_size_train, shuffle=True, drop_last=False)
    eval_loader = DataLoader(Satellite(partition='eval', scalar=args.scalar, gaussian_noise=False),
                             batch_size=args.batch_size_val, shuffle=False, drop_last=False)
    test_loader = DataLoader(Satellite(partition='test', scalar=args.scalar, gaussian_noise=False),
                             batch_size=args.batch_size_test, shuffle=False, drop_last=False)

    boardio = SummaryWriter(logdir='checkpoints/deepcpd')
    textio = open('checkpoints/deepcpd/run.log', 'a')
    textio.write(str(args) + '\n')
    textio.flush()
    #########################################################
    ## Train and Test Model
    #########################################################
    deepcpd = DeepCPD(args)

    if args.eval:
        # Testing Phase
        total_loss, angles_error, translations_error, time_costs = deepcpd.test(test_loader)
        print('Average_rotation_error:', angles_error, '\n', 'Average_translations_error:',
              translations_error * args.scalar, '\n', 'Average_time_costs:', time_costs)
    else:
        # Training Phase
        deepcpd.train(train_loader, eval_loader, textio, boardio)

