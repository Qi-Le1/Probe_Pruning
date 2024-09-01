import errno
import numpy as np
import io
import os
import pickle
import torch
from torchvision.utils import save_image
from .utils import recur
from config import cfg



def check_calib_saving_info():
    current_script_dir = os.path.dirname(__file__)
    result_path = os.path.join(current_script_dir, '..', 'output', 'result', 'calibsavinginfo')
    name_list = cfg['model_tag'].split('_')
    # data_name
    name_list[1] = 'None'
    # task_name
    name_list[3] = 'None'
    # batch_size
    name_list[4] = 'None'
    # prune_ratio
    name_list[6] = 'None'
    # prune_method
    name_list[8] = 'None' if 'skip' not in cfg['model_tag'] else name_list[8]
    # mode
    name_list[9] = 'None'
    # prune_info
    name_list[11] = 'None'
    # cust_tgt_modules
    # name_list[12] = 'None'
    calibsavinginfo_path = os.path.join(result_path, '_'.join(name_list))
    if os.path.exists(calibsavinginfo_path) and load(calibsavinginfo_path, mode='torch') is not None:
        return True
    return False


def load_calib_saving_info(model):
    # variable: model_name, max_seq_len, prune_metric, calib_info, cust_tgt_modules
    current_script_dir = os.path.dirname(__file__)
    result_path = os.path.join(current_script_dir, '..', 'output', 'result', 'calibsavinginfo')
    name_list = cfg['model_tag'].split('_')
    # data_name
    name_list[1] = 'None'
    # task_name
    name_list[3] = 'None'
    # batch_size
    name_list[4] = 'None'
    # prune_ratio
    name_list[6] = 'None'
    # prune_method
    name_list[8] = 'None' if 'skip' not in cfg['model_tag'] else name_list[8]
    # mode
    name_list[9] = 'None'
    # prune_info
    name_list[11] = 'None'
    # cust_tgt_modules
    # name_list[12] = 'None'
    calibsavinginfo_path = os.path.join(result_path, '_'.join(name_list))
    
    # Load the calibration data
    calibration_data = load(calibsavinginfo_path, mode='torch')

    # Apply the loaded calibration data to the model
    for key, value in calibration_data.items():
        module_name, attr_name = key.rsplit('+', 1)
        print(f"Applying calibration data to {module_name}.{attr_name}...")
        module = dict(model.named_modules())[module_name]
        module_device = module.weight.device

        # Move the value to the same device as the module's weights
        value_to_device = value.to(module_device)
        
        # Set the attribute with the value that's now on the correct device
        setattr(module, attr_name, value_to_device)
    print("Calibration data applied to the model successfully.")
    return



def save_calib_info(model):
    current_script_dir = os.path.dirname(__file__)
    result_path = os.path.join(current_script_dir, '..', 'output', 'result', 'calibsavinginfo')
    makedir_exist_ok(result_path)
    name_list = cfg['model_tag'].split('_')
    # data_name
    name_list[1] = 'None'
    # task_name
    name_list[3] = 'None'
    # batch_size
    name_list[4] = 'None'
    # prune_ratio
    name_list[6] = 'None'
    # prune_method
    name_list[8] = 'None' if 'skip' not in cfg['model_tag'] else name_list[8]
    # mode
    name_list[9] = 'None'
    # prune_info
    name_list[11] = 'None'
    # cust_tgt_modules
    # name_list[12] = 'None'
    calibsavinginfo_path = os.path.join(result_path, '_'.join(name_list))
    
    all_calin_info = {}
    for name, module in model.named_modules():
        if hasattr(module, 'return_global_metric_info'):
            data = module.return_global_metric_info()
            if data is not None:
                for key, value in data.items():
                    all_calin_info[f"{name}+{key}"] = value
            
    save(all_calin_info, calibsavinginfo_path, mode='torch')
    print("Calibration data saved successfully.")
    return
    

def check_dense_model():
    current_script_dir = os.path.dirname(__file__)
    result_path = os.path.join(current_script_dir, '..', 'output', 'result')
    dense_name_list = cfg['model_tag'].split('_')
    # batch_size
    dense_name_list[4] = str(cfg[cfg['model_name']]['batch_size']['test'])
    # prune_ratio
    dense_name_list[6] = '0'
    # prune_metric
    dense_name_list[7] = 'None'
    # prune_method
    dense_name_list[8] = 'dense'
    # mode
    dense_name_list[9] = 'None'
    # calib_info
    dense_name_list[10] = 'None'
    # prune_info
    dense_name_list[11] = 'None'
    # cust_tgt_modules
    dense_name_list[12] = 'None'
    dense_model_path = os.path.join(result_path, '_'.join(dense_name_list))
    if not os.path.exists(dense_model_path):
        dense_model_path = os.path.join(result_path, 'dense', '_'.join(dense_name_list))
        if not os.path.exists(dense_model_path):
            return None
        else:
            return dense_model_path
    else:
        return dense_model_path
    

def load_dense_model():    
    from .io import load
    dense_model_path = check_dense_model()
    if dense_model_path is None:
        return None, None
    dense_res = load(dense_model_path)
    dense_info_list, dense_duration = dense_res['dense_info_list'], dense_res['dense_duration']
    return dense_info_list, dense_duration

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
    return None

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
    return None


# def save(input, path, mode='pickle'):
#     dirname = os.path.dirname(path)
#     makedir_exist_ok(dirname)
#     if mode == 'torch':
#         torch.save(input, path)
#     elif mode == 'np':
#         np.save(path, input, allow_pickle=True)
#     elif mode == 'pickle':
#         pickle.dump(input, open(path, 'wb'))
#     else:
#         raise ValueError('Not valid save mode')
#     return None

# class CPU_Unpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         if module == 'torch.storage' and name == '_load_from_bytes':
#             return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
#         else: return super().find_class(module, name)

# def load(path, mode='pickle'):
#     if not torch.cuda.is_available() and mode == 'pickle':
#         return CPU_Unpickler(open(path, 'rb')).load()
#     if mode == 'torch':
#         return torch.load(path, map_location=lambda storage, loc: storage)
#     elif mode == 'np':
#         return np.load(path, allow_pickle=True)
#     elif mode == 'pickle':
#         return pickle.load(open(path, 'rb'))
#     else:
#         raise ValueError('Not valid save mode')
#     return None


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
