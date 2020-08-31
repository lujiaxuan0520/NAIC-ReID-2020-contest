from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pclreid import Pclreid

# the map between given dataset name and actual name
__imgreid_factory = {
    'pclreid': Pclreid,
}


def get_names():
    return list(__imgreid_factory.keys())


def init_imgreid_dataset(name, **kwargs):
    if name not in list(__imgreid_factory.keys()):
        raise KeyError("Invalid dataset, got '{}', but expected to be one of {}".format(name, list(__imgreid_factory.keys())))
    return __imgreid_factory[name](**kwargs) # get the dataset