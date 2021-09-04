mpiexec -n 4 --hostfile hosts  --oversubscribe  python /home/pi/PyPi/chainer_cifar_multi_node.py

mpiexec -n 2  python3 /Users/kahira/Desktop/PyPi/chainer/train_data_parallel.py

mpiexec -n 2  python3 /Users/kahira/Desktop/PyPi/chainer/train_model_parallel.py
