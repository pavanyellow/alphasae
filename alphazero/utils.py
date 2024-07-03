
from numbers import Number
import torch
from dataclasses import dataclass
from sae.model import SparseAutoencoder
import sys
from collections.abc import Mapping, Container
import numpy as np

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]


@dataclass
class NNetWrapperConfig:
    lr : float = 0.001
    dropout : float = 0.3
    epochs : int = 10
    batch_size : int = 256
    cuda : bool = torch.cuda.is_available()
    num_channels : int = 512
    sae : SparseAutoencoder = None
    layer_num : int = 0
    collect_resid_activations : bool = False
    collect_sae_feature_activations : bool = False
    replace_sae_activations : bool = False

class Activations:
    def __init__(self, num_layers = 2):
        self.neurons = [[] for _ in range(num_layers)]
        self.boards = []
        self.features = [[] for _ in range(num_layers)]
        self.current_boards = set()
    
    def clear(self):
        self.neurons = [[] for _ in range(len(self.neurons))]
        self.features = [[] for _ in range(len(self.features))]
    


def deep_getsizeof(obj, seen=None):
    """Recursively calculate size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, (str, bytes, Number, range, bytearray)):
        pass
    elif isinstance(obj, (tuple, list, set, frozenset)):
        size += sum(deep_getsizeof(i, seen) for i in obj)
    elif isinstance(obj, Mapping):
        size += sum(deep_getsizeof(k, seen) + deep_getsizeof(v, seen) for k, v in obj.items())
    elif hasattr(obj, '__dict__'):
        size += deep_getsizeof(obj.__dict__, seen)
    return size

def print_size(obj, name):
    """Print size of object in KB or MB"""
    size_bytes = deep_getsizeof(obj)
    if size_bytes < 1024 * 1024:
        print(f"{name}: {size_bytes / 1024:.2f} KB")
    else:
        print(f"{name}: {size_bytes / (1024 * 1024):.2f} MB")

def print_size_of(mcts):
    """Print sizes of MCTS attributes"""
    print_size(mcts, "Entire MCTS")
    print_size(mcts.Qsa, "Qsa")
    print_size(mcts.Nsa, "Nsa")
    print_size(mcts.Ns, "Ns")
    print_size(mcts.Ps, "Ps")
    print_size(mcts.Es, "Es")
    print_size(mcts.Vs, "Vs")
