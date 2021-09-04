#!/usr/bin/env python
import argparse
import chainer
import time
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import get_cifar10
import chainermn
import numpy as np

from models.alexnet_model_parallel import AlexNet
from config import config



def main():
    comm = chainermn.create_communicator("naive")

    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--device', '-d', type=str, default='-1',
                        help='Device specifier. Either ChainerX device '
                        'specifier or an integer. If non-negative integer, '
                        'CuPy arrays with specified device id are used. If '
                        'negative integer, NumPy arrays are used')
    parser.add_argument('--snapshot', '-s',
                        default='result/snapshot_iter_12000',
                        help='The path to a saved snapshot (NPZ)')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    group = parser.add_argument_group('deprecated arguments')
    group.add_argument('--gpu', '-g', dest='device',
                       type=int, nargs='?', const=0,
                       help='GPU ID (negative value indicates CPU)')
    args = parser.parse_args()

    device = chainer.get_device(args.device)

    print('Device: {}'.format(device))
    print('# unit: {}'.format(args.unit))
    print('')

    device.use()

    # Create a same model object as what you used for training
    model = AlexNet(comm, num_classes=10)
    optimizer = chainer.optimizers.MomentumSGD(0.001)
    optimizer.setup(model)

    # # Load saved parameters from a NPZ file of the Trainer object
    # try:
    #     chainer.serializers.load_npz(
    #         "result/snapshot_iter_12000", model, path='updater/model:main/predictor/')
    # except Exception:
    #     chainer.serializers.load_npz(
    #         args.snapshot, model, path='predictor/')

    #model.to_device(device)

    # Prepare data
    train, test = get_cifar10()
    x = test.__getitem__(0)[0]
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    with chainer.using_config('train', False):
        prediction = model.forward(x)

    print('Prediction:', prediction)
    #print('Answer:', answer)


if __name__ == '__main__':
    main()
