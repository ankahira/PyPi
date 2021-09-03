#!/usr/bin/env python

from __future__ import print_function
import argparse

import chainer.backends.cuda
from chainer import training
from chainer.training import extensions

import chainermn
import chainermnx

import chainer.links as L

# Local Imports
from models.alexnet_filter import AlexNet


def main():

    parser = argparse.ArgumentParser(description='Train Cifar Filter Parallel')
    parser.add_argument('--batchsize', '-B', type=int, default=32, help='Learning minibatch size')
    parser.add_argument('--epochs', '-E', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--out', '-o', default='results', help='Output directory')
    args = parser.parse_args()

    batch_size = args.batchsize
    epochs = args.epochs
    out = args.out



    # Prepare ChainerMN communicator.
    comm = chainermnx.create_communicator('naive_channel')
    device = -1

    if comm.rank == 0:
        print('==========================================')
        print('Num of GPUs : {}'.format(comm.size))
        print('Minibatch-size: {}'.format(batch_size))
        print('Epochs: {}'.format(args.epochs))
        print('==========================================')

    # model = L.Classifier(models[args.model](comm))

    train, test = chainer.datasets.get_cifar10()
    model = L.Classifier(AlexNet(comm))

    # Split and distribute the dataset. Only worker 0 loads the whole dataset.
    # Datasets of worker 0 are evenly split and distributed to all workers.


    # Create a multinode iterator such that each rank gets the same batch
    if comm.rank != 0:
        train = chainermn.datasets.create_empty_dataset(train)
        val = chainermn.datasets.create_empty_dataset(test)
    # Same dataset in all nodes
    train_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(train, batch_size, n_threads=20), comm)
    val_iter = chainermn.iterators.create_multi_node_iterator(
        chainer.iterators.MultithreadIterator(val, batch_size, n_threads=20, repeat=False), comm)

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Set up a trainer
    updater = chainermnx.training.StandardUpdater(train_iter, optimizer, comm, out=out, device=device)
    trainer = training.Trainer(updater, (epochs, 'iteration'), out)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    # Create an evaluator
    evaluator = extensions.Evaluator(val_iter, model, device=device)
    trainer.extend(evaluator, trigger=val_interval)

    # Some display and output extensions are necessary only for one worker.

    # Some display and output extensions are necessary only for one worker.
    if comm.rank == 0:
        # trainer.extend(extensions.DumpGraph('main/loss'))
        trainer.extend(extensions.LogReport(trigger=log_interval))
        trainer.extend(extensions.observe_lr(), trigger=(1, 'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=10))
        trainer.extend(extensions.PrintReport(
            ['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        trainer.extend(extensions.PlotReport(
            ['main/loss', 'validation/main/loss'], 'epoch', filename='loss.png'))
        trainer.extend(extensions.PlotReport(
            ['main/accuracy', 'validation/main/accuracy'], 'epoch', filename='accuracy.png'))
        trainer.extend(extensions.ProgressBar())
    # TODO : Figure out how to send this report to a file

    if comm.rank == 0:
        print("Starting training .....")

    # hook = TimerHook()
    # with hook:
    trainer.run()
    if comm.rank == 0:
        print("Finished")

    # if comm.rank == 0:
    #     hook.print_report()


if __name__ == '__main__':
    main()
