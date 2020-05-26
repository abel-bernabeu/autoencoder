import sys
from argparse import ArgumentParser
import torch as torch

def addTrainablesArg(parser):
    parser.add_argument(
        '--trainables', dest='trainables', help='Trainable parameters directory')

def addExchangeArg(parser):
    parser.add_argument(
        '--exchange', dest='exchange', help='File with exchanged data')

parser = ArgumentParser()

subparsers = parser.add_subparsers(dest="action")

train_parser = subparsers.add_parser('train')
train_parser.add_argument(
  '--train-samples-percentage', dest='train_samples_percentage', help='Percentage of samples used for training')
train_parser.add_argument(
  '--batch-size', dest='batch_size', help='Batch size')
train_parser.add_argument(
  '--epochs', dest='epochs', help='Number of epochs')
addTrainablesArg(train_parser)

encode_parser = subparsers.add_parser('encode')
addTrainablesArg(encode_parser)
encode_parser.add_argument(
  '--input-image', dest='input_image', help='Input image file name')
addExchangeArg(encode_parser)

decode_parser = subparsers.add_parser('decode')
addTrainablesArg(decode_parser)
addExchangeArg(decode_parser)
decode_parser.add_argument(
  '--output-image', dest='output_image', help='Output image file name')

opts = parser.parse_args()

if opts.action == 'train':
    if (not torch.cuda.is_available()):
        print('Error: CUDA is needed for training')
        sys.exit(0)
    print ('TODO: training')
elif opts.action == 'encode':
    print ('TODO: encoding')
elif opts.action == 'decode':
    print ('TODO: decoding')