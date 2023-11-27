from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR10, CIFAR100
from .femnist import FEMNIST
from .svhn import SVHN
from .utils import *

__all__ = [
    'MNIST', 
    'FashionMNIST', 
    'FEMNIST',
    'CIFAR10', 
    'CIFAR100', 
    'SVHN'
]
