import argparse
import models



parser = argparse.ArgumentParser(description='PyTorch MedMNIST Training')
parser.add_argument('--seed', type = int ,default=0,help='seed')
parser.add_argument('--dataset', default='medmnist', help='dataset setting')
parser.add_argument('--data_flag', default='pneumoniamnist', help='medmnist dataset')
parser.add_argument('--classes', default=2, type=int, help='number of classes ')
parser.add_argument('--num_in_channels', default=1, type=int, help='input channels 1 or 3 for medmnist ')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='mlp')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                help='number of data loading workers (default: 4)')
parser.add_argument('--max_epoch', default=10**4, type=int, metavar='N',
                help='number of maximum epochs to run')
parser.add_argument('-b', '--batch-size', default=128*4, type=int,
                metavar='N',
                help='mini-batch size')
parser.add_argument('--train_sample_size', type = int ,default=4708,help='test_size')
parser.add_argument('--val_sample_size', type = int ,default=524,help='test_size')
parser.add_argument('--test_size', type = float ,default=0.3,help='test_size')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--gamma', type = float, default=0.2995,help='gamma')
parser.add_argument('--gpu', default=0,help='device')
parser.add_argument('--root_log',type=str, default='medmnist_log')
parser.add_argument('--root_model', type=str, default='medmnist_checkpoint')