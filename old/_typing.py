from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    Hashable,
    Iterator,
    Literal,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    TypeVar,
    Union,
    List,
    Dict
)

if TYPE_CHECKING:
    from datasets import (
        # MNIST,
        # FashionMNIST,
        FashionMNIST,
        CIFAR10,
        CIFAR100,
        # SVHN
    )
    # from datasets.mnist import MNIST
    # from datasets.mnist import FashionMNIST
    # from datasets.cifar import CIFAR10
    # from datasets.cifar import CIFAR100
    # from datasets.svhn import SVHN

    from models.api import (
        ResNet,
        WideResNet
    )
    # from models.resnet import ResNet
    # from models.wresnet import WideResNet

    from metrics.metrics import Metric

    from logger import Logger

    from modules.api import (
        Client,
    )
    
    from modules.server.api import (
        ServerDynamicFL,
        ServerFedAvg,
        ServerFedEnsemble,
        ServerFedGen,
        ServerFedProxy,
        ServerDynamicSgd,
        ServerDynamicAvg,
        ServerCombinationSearch
    )

    from modules.client.api import (
        ClientDynamicFL,
        ClientFedAvg,
        ClientFedGen,
        ClientFedProxy,
        ClientDynamicSgd,
        ClientDynamicAvg
    )



from torch.utils.data import DataLoader

from torch.optim import (
        SGD,
        Adam,
        LBFGS
    )

from torch.optim.lr_scheduler import (
    MultiStepLR,
    StepLR,
    ExponentialLR,
    CosineAnnealingLR,
    ReduceLROnPlateau,
    CyclicLR
)

DatasetType = Any
# Union[ 
    # type['MNIST'],
    # type['FashionMNIST'],
    # type['CIFAR10'],
    # type['CIFAR100'],
    # type['SVHN']
# ]

OptimizerType = Union[
    type[SGD],
    type[Adam],
    type[LBFGS]
]

SchedulerType = Union[
    type[MultiStepLR],
    type[StepLR],
    type[ExponentialLR],
    type[CosineAnnealingLR],
    type[ReduceLROnPlateau],
    type[CyclicLR]
]

DataLoaderType = type[DataLoader]

ModelType = Union[
    type['WideResNet'],
    type['ResNet']
]

MetricType = type['Metric']

LoggerType = type['Logger']

ClientType = type['Client']

ServerType = Union[
    type['ServerDynamicFL'],
    type['ServerFedAvg'],
    type['ServerFedEnsemble'],
    type['ServerFedGen'],
    type['ServerFedProxy'],
    type['ServerDynamicSgd'],
    type['ServerDynamicAvg'],
    type['ServerCombinationSearch']
]

ClientType = Union[
    type['ClientDynamicFL'],
    type['ClientFedAvg'],
    type['ClientFedGen'],
    type['ClientFedProxy'],
    type['ClientDynamicSgd'],
    type['ClientDynamicAvg']
]

Tag = Literal[
    'client',
    'server'
]

Local_Gradient_Update = int