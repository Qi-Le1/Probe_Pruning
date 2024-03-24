import errno
import numpy as np
import io
import os
import pickle
import torch
from torchvision.utils import save_image
from .utils import recur


def remove_non_picklable_items(input_dict):
    non_picklable_keys = []
    for key, value in input_dict.items():
        try:
            pickle.dumps(value)
        except (pickle.PicklingError, TypeError):
            non_picklable_keys.append(key)

    # Remove the non-picklable items
    for key in non_picklable_keys:
        del input_dict[key]

    return

def check_exists(path):
    return os.path.exists(path)


def makedir_exist_ok(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise
    return

# def save(input, path, mode='torch'):
#     dirname = os.path.dirname(path)
#     makedir_exist_ok(dirname)
#     if mode == 'torch':
#         torch.save(input, path, pickle_protocol=4)
#     elif mode == 'np':
#         np.save(path, input, allow_pickle=True)
#     elif mode == 'pickle':
#         pickle.dump(input, open(path, 'wb'))
#     else:
#         raise ValueError('Not valid save mode')
#     return

# def load(path, mode='torch'):
#     if mode == 'torch':
#         return torch.load(path, map_location=lambda storage, loc: storage)
#     elif mode == 'np':
#         return np.load(path, allow_pickle=True)
#     elif mode == 'pickle':
#         return pickle.load(open(path, 'rb'))
#     else:
#         raise ValueError('Not valid save mode')
#     return

def save(input, path, mode='pickle'):
    dirname = os.path.dirname(path)
    makedir_exist_ok(dirname)
    if mode == 'torch':
        torch.save(input, path)
    elif mode == 'np':
        np.save(path, input, allow_pickle=True)
    elif mode == 'pickle':
        pickle.dump(input, open(path, 'wb'))
    else:
        raise ValueError('Not valid save mode')
    return

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def load(path, mode='pickle'):
    if not torch.cuda.is_available() and mode == 'pickle':
        return CPU_Unpickler(open(path, 'rb')).load()
    if mode == 'torch':
        return torch.load(path, map_location=lambda storage, loc: storage)
    elif mode == 'np':
        return np.load(path, allow_pickle=True)
    elif mode == 'pickle':
        return pickle.load(open(path, 'rb'))
    else:
        raise ValueError('Not valid save mode')
    return


def save_img(img, path, nrow=10, padding=1, pad_value=0, value_range=None):
    makedir_exist_ok(os.path.dirname(path))
    normalize = False if range is None else True
    save_image(img, path, nrow=nrow, padding=padding, pad_value=pad_value, normalize=normalize, value_range=value_range)
    return


def to_device(input, device):
    output = recur(lambda x, y: x.to(y), input, device)
    return output


def resume(path, verbose=True, resume_mode=1):
    if os.path.exists(path) and resume_mode == 1:
        result = load(path)
        if verbose:
            print('Resume from {}'.format(result['epoch']))
    else:
        if resume_mode == 1:
            print('Not exists: {}'.format(path))
        result = None
    return result
