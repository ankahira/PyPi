import argparse
import chainer
import time
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import get_cifar10
import chainermn

from models.alexnet import AlexNet
from config import config


def main():
    chainer.global_config.cudnn_deterministic = True

    # Prepare ChainerMN communicator.
    comm = chainermn.create_communicator("naive")
    if comm.rank == 0:
        print('==========================================')
        print('Num process (COMM_WORLD): {}'.format(comm.size))
        print('Num Minibatch-size: {}'.format(config.BATCH_SIZE))
        print('Num epoch: {}'.format(config.EPOCHS))
        print('==========================================')

    # Set up a neural network to train.
    # Classifier reports softmax cross entropy loss and accuracy at every
    # iteration, which will be used by the PrintReport extension below.
    if comm.rank == 0:
        train, test = get_cifar10()
    else:
        train, test = None, None

    train = chainermn.scatter_dataset(train, comm, shuffle=False)
    test = chainermn.scatter_dataset(test, comm, shuffle=False)

    model = L.Classifier(AlexNet(num_classes=10))

    optimizer = chainer.optimizers.MomentumSGD(lr=config.LEARNING_RATE, momentum=config.MOMENTUM)
    optimizer = chainermn.create_multi_node_optimizer(optimizer, comm)
    optimizer.setup(model)

    train_iter = chainer.iterators.SerialIterator(train, config.BATCH_SIZE,   shuffle=True)
    test_iter = chainer.iterators.SerialIterator(test, config.BATCH_SIZE,  repeat=False, shuffle=True)

    # Set up a trainer
    updater = training.updaters.StandardUpdater(train_iter, optimizer)
    trainer = training.Trainer(updater, (config.EPOCHS, 'epoch'), out=config.output_dir)

    val_interval = (1, 'epoch')
    log_interval = (1, 'epoch')

    # Evaluate the model with the test dataset for each epoch
    evaluator = extensions.Evaluator(test_iter, model)
    evaluator = chainermn.create_multi_node_evaluator(evaluator, comm)
    trainer.extend(evaluator, trigger=val_interval)
    if comm.rank == 0:
        trainer.extend(extensions.dump_graph('main/loss'))
        trainer.extend(extensions.LogReport())
        trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
             'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
        if config.CHECKPOINT:
            trainer.extend(extensions.snapshot(filename='snaphot_epoch_{.updater.epoch}'), trigger=(5, 'epoch'))
        trainer.extend(extensions.ProgressBar(update_interval=1))

    if config.RESUME:
        # Resume from a snapshot
        chainer.serializers.load_npz("checkpoint.npz", trainer)

    # Run the training
    start = time.time()
    trainer.run()
    stop = time.time()
    if comm.rank == 0:
        print("Total Training  Time", stop - start)


if __name__ == '__main__':
    main()
