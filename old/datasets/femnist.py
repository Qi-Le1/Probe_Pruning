import numpy as np
import os
import pickle
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.api import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts, load_obj, save_obj, create_folder

class FEMNIST(Dataset):
    data_name = 'FEMNIST'
    file = [('https://s3.amazonaws.com/nist-srd/SD19/by_write.zip', None), 
            ('https://s3.amazonaws.com/nist-srd/SD19/by_class.zip', None)]

    def __init__(self, root, split, transform=None):

        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        # train_set, test_set, meta = self.make_data()
        # save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        # save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        # save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        # print(self.processed_folder)
        if not check_exists(self.processed_folder):
            self.process()
        
        # parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        # by_writer_dir = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'images_by_writer')
        # writers = load_obj(by_writer_dir)

       
        self.id, self.data, self.target = load(os.path.join(self.processed_folder, '{}.pt'.format(self.split)),
                                               mode='pickle')
        
        # self.id = self.id[:int(len(self.id)/5000)]
        # self.data = self.data[:int(len(self.data)/5000)]
        # self.target = self.target[:int(len(self.target)/5000)]

        self.classes_counts = make_classes_counts(self.target)
        self.classes_to_labels, self.target_size = load(os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        a = 5
        
    def __getitem__(self, index):
        id, data, target = torch.tensor(self.id[index]), Image.fromarray(self.data[index]), torch.tensor(
            self.target[index], dtype=torch.int64)
        input = {'id': id, 'data': data, 'target': target}
        if self.transform is not None:
            input = self.transform(input)
        return input

    def __len__(self):
        return len(self.data)

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def process(self):
        if not check_exists(self.raw_folder):
            self.download()

        get_files_dirs()
        get_hashes()
        match_hashes()
        group_by_writer()

        # data_to_json()
        train_set, test_set, meta = self.make_data()
        save(train_set, os.path.join(self.processed_folder, 'train.pt'), mode='pickle')
        save(test_set, os.path.join(self.processed_folder, 'test.pt'), mode='pickle')
        save(meta, os.path.join(self.processed_folder, 'meta.pt'), mode='pickle')
        return

    def download(self):
        makedir_exist_ok(self.raw_folder)
        for (url, _) in self.file:
            filename = os.path.basename(url)
            download_url(url, os.path.join(self.raw_folder, filename))
            extract_file(os.path.join(self.raw_folder, filename))
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        import math
        import json
        
        def relabel_class(c):
            '''
            maps hexadecimal class value (string) to a decimal number
            returns:
            - 0 through 9 for classes representing respective numbers
            - 10 through 35 for classes representing respective uppercase letters
            - 36 through 61 for classes representing respective lowercase letters
            '''
            if c.isdigit() and int(c) < 40:
                return (int(c) - 30)
            elif int(c, 16) <= 90: # uppercase
                return (int(c, 16) - 55)
            else:
                return (int(c, 16) - 61)

        # def relabel_class_once_more(c):
        #     '''
        #     maps decimal number to 0 - 49
        #     returns:
        #     - 0 through 9 for classes representing respective numbers
        #     - 10 through 35 for classes representing respective uppercase letters
        #     - 36 through 61 for classes representing respective lowercase letters
        #     '''
        #     if 10 <= c <= 34: # uppercase
        #         return c - 10
        #     elif 36 <= c <= 60: # lowercase
        #         return c - 11
        #     else:
        #         raise ValueError('class value {} is not in the range 10-34 or 36-60'.format(c))
            
        parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

        by_writer_dir = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'images_by_writer')
        writers = load_obj(by_writer_dir)
        MAX_WRITERS = len(writers)  # max number of writers per json file.

        num_json = int(math.ceil(len(writers) / MAX_WRITERS))

        users = []
        num_samples = []
        user_data = {}
        all_data = []
        all_labels = []
        writer_count, all_writers = 0, 0
        json_index = 0
        for (w, l) in writers:
            
            users.append(w)
            num_samples.append(len(l))
            user_data[w] = {'x': [], 'y': []}

            size = 28, 28  # original image size is 128, 128
            label_kinds = set()
            for (f, c) in l:
                file_path = os.path.join(parent_path, f)
                img = Image.open(file_path)
                gray = img.convert('L')
                gray.thumbnail(size, Image.ANTIALIAS)
                arr = np.asarray(gray).copy()
                vec = arr
                # vec = arr.flatten()
                vec = vec / 255  # scale all pixel values to between 0 and 1
                vec = vec.tolist()

                nc = relabel_class(c)

                # if nc in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 35, 61]:
                #     continue
                # # user_data[w]['x'].append(vec)
                # # user_data[w]['y'].append(nc)
                # # num_samples += 1
                # nc = relabel_class_once_more(nc)
                all_data.append(vec)
                all_labels.append(nc)
                label_kinds.add(nc)

            writer_count += 1
            all_writers += 1
            if all_writers % 50 == 0:
                print(f'all_writers: {all_writers}\n')
                # if all_writers == 100:
                #     break
            # if writer_count == MAX_WRITERS or all_writers == len(writers):

            #     all_data = {}
            #     all_data['users'] = users
            #     all_data['num_samples'] = num_samples
            #     all_data['user_data'] = user_data

            #     file_name = 'all_data_%d.json' % json_index
            #     file_path = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', file_name)

            #     print('writing %s' % file_name)

            #     with open(file_path, 'w') as outfile:
            #         json.dump(all_data, outfile)

            #     writer_count = 0
            #     json_index += 1
                
            #     users[:] = []
            #     num_samples[:] = []
            #     user_data.clear()
        # train_filenames = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
        # test_filenames = ['test_batch']
        # train_data, train_target = read_pickle_file(
        #     os.path.join(self.raw_folder, 'cifar-10-batches-py'),
        #     train_filenames
        # )
        # test_data, test_target = read_pickle_file(
        #     os.path.join(self.raw_folder, 'cifar-10-batches-py'), 
        #     test_filenames
        # )
        total_sample = sum(num_samples)
        print(f'total_sample: {total_sample}\n')

        all_data = np.array(all_data)
        all_labels = np.array(all_labels)
        total_samples_num = len(all_labels)
        print(f'total_samples_num_need: {total_samples_num}\n')
        all_ravel = all_data.ravel()
        all_mean = np.mean(all_ravel)
        all_std = np.std(all_ravel)
        print(f'femnist_mean: {all_mean}')
        print(f'femnist_std: {all_std}')
        idx = np.random.permutation(total_samples_num)
        num_train = int(total_samples_num * 0.8)
        
        train_idx, test_idx = idx[:num_train], idx[num_train:]
        train_data, train_target = all_data[train_idx], all_labels[train_idx]
        test_data, test_target = all_data[test_idx], all_labels[test_idx]

        train_id, test_id = np.arange(len(train_data)).astype(np.int64), np.arange(len(test_data)).astype(np.int64)
        # with open(os.path.join(self.raw_folder, 'cifar-10-batches-py', 'batches.meta'), 'rb') as f:
        #     data = pickle.load(f, encoding='latin1')
        #     classes = data['label_names']
        unique_class_num = len(label_kinds)
        print(f'unique_class_num: {unique_class_num}\n')
        classes_to_labels = {i: i for i in range(unique_class_num)}
        target_size = unique_class_num

        return (train_id, train_data, train_target), (test_id, test_data, test_target), (classes_to_labels, target_size)



def get_files_dirs():

    import os
    import sys

    # utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # utils_dir = os.path.join(utils_dir, 'utils')
    # sys.path.append(utils_dir)
    # import utils
    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    create_folder(os.path.join(parent_path, 'data', 'FEMNIST', 'processed'))
    print('parent_path', parent_path)
    class_files = []  # (class, file directory)
    write_files = []  # (writer, file directory)

    class_dir = os.path.join(parent_path, 'data', 'FEMNIST', 'raw', 'by_class')
    rel_class_dir = os.path.join('data', 'FEMNIST', 'raw', 'by_class')
    classes = os.listdir(class_dir)
    classes = [c for c in classes if len(c) == 2]

    for cl in classes:
        cldir = os.path.join(class_dir, cl)
        rel_cldir = os.path.join(rel_class_dir, cl)
        subcls = os.listdir(cldir)

        subcls = [s for s in subcls if (('hsf' in s) and ('mit' not in s))]

        for subcl in subcls:
            subcldir = os.path.join(cldir, subcl)
            rel_subcldir = os.path.join(rel_cldir, subcl)
            images = os.listdir(subcldir)
            image_dirs = [os.path.join(rel_subcldir, i) for i in images]

            for image_dir in image_dirs:
                class_files.append((cl, image_dir))

    write_dir = os.path.join(parent_path, 'data', 'FEMNIST', 'raw', 'by_write')
    rel_write_dir = os.path.join('data', 'FEMNIST', 'raw', 'by_write')
    write_parts = os.listdir(write_dir)

    for write_part in write_parts:
        writers_dir = os.path.join(write_dir, write_part)
        rel_writers_dir = os.path.join(rel_write_dir, write_part)
        writers = os.listdir(writers_dir)

        for writer in writers:
            writer_dir = os.path.join(writers_dir, writer)
            rel_writer_dir = os.path.join(rel_writers_dir, writer)
            wtypes = os.listdir(writer_dir)

            for wtype in wtypes:
                type_dir = os.path.join(writer_dir, wtype)
                rel_type_dir = os.path.join(rel_writer_dir, wtype)
                images = os.listdir(type_dir)
                image_dirs = [os.path.join(rel_type_dir, i) for i in images]

                for image_dir in image_dirs:
                    write_files.append((writer, image_dir))

    save_obj(
        class_files,
        os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'class_file_dirs'))
    save_obj(
        write_files,
        os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_file_dirs'))
    return


def get_hashes():
    import hashlib
    import os
    import sys

    # utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # utils_dir = os.path.join(utils_dir, 'utils')
    # sys.path.append(utils_dir)

    # import util

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    cfd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'class_file_dirs')
    wfd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_file_dirs')
    class_file_dirs = load_obj(cfd)
    write_file_dirs = load_obj(wfd)

    class_file_hashes = []
    write_file_hashes = []

    count = 0
    for tup in class_file_dirs:
        if (count % 100000 == 0):
            print('hashed %d class images' % count)

        (cclass, cfile) = tup
        file_path = os.path.join(parent_path, cfile)

        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        class_file_hashes.append((cclass, cfile, chash))

        count += 1

    cfhd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'class_file_hashes')
    save_obj(class_file_hashes, cfhd)

    count = 0
    for tup in write_file_dirs:
        if (count % 100000 == 0):
            print('hashed %d write images' % count)

        (cclass, cfile) = tup
        file_path = os.path.join(parent_path, cfile)

        chash = hashlib.md5(open(file_path, 'rb').read()).hexdigest()

        write_file_hashes.append((cclass, cfile, chash))

        count += 1

    wfhd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_file_hashes')
    save_obj(write_file_hashes, wfhd)

    return

def match_hashes():
    import os
    import sys

    # utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # utils_dir = os.path.join(utils_dir, 'utils')
    # sys.path.append(utils_dir)

    # import util

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    cfhd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'class_file_hashes')
    wfhd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_file_hashes')
    class_file_hashes = load_obj(cfhd) # each elem is (class, file dir, hash)
    write_file_hashes = load_obj(wfhd) # each elem is (writer, file dir, hash)

    class_hash_dict = {}
    for i in range(len(class_file_hashes)):
        (c, f, h) = class_file_hashes[len(class_file_hashes)-i-1]
        class_hash_dict[h] = (c, f)

    write_classes = []
    for tup in write_file_hashes:
        (w, f, h) = tup
        write_classes.append((w, f, class_hash_dict[h][0]))

    wwcd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_with_class')
    save_obj(write_classes, wwcd)

    return

def group_by_writer():
    import os
    import sys

    # utils_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    # utils_dir = os.path.join(utils_dir, 'utils')
    # sys.path.append(utils_dir)

    # import util

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    wwcd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'write_with_class')
    write_class = load_obj(wwcd)

    writers = [] # each entry is a (writer, [list of (file, class)]) tuple
    cimages = []
    (cw, _, _) = write_class[0]
    for (w, f, c) in write_class:
        if w != cw:
            writers.append((cw, cimages))
            cw = w
            cimages = [(f, c)]
        cimages.append((f, c))
    writers.append((cw, cimages))

    ibwd = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'images_by_writer')
    save_obj(writers, ibwd)
    return

def data_to_json():
    import math
    import json
    MAX_WRITERS = 100  # max number of writers per json file.

    def relabel_class(c):
        '''
        maps hexadecimal class value (string) to a decimal number
        returns:
        - 0 through 9 for classes representing respective numbers
        - 10 through 35 for classes representing respective uppercase letters
        - 36 through 61 for classes representing respective lowercase letters
        '''
        if c.isdigit() and int(c) < 40:
            return (int(c) - 30)
        elif int(c, 16) <= 90: # uppercase
            return (int(c, 16) - 55)
        else:
            return (int(c, 16) - 61)

    parent_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

    by_writer_dir = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', 'images_by_writer')
    writers = load_obj(by_writer_dir)

    num_json = int(math.ceil(len(writers) / MAX_WRITERS))

    users = []
    num_samples = []
    user_data = {}

    writer_count, all_writers = 0, 0
    json_index = 0
    for (w, l) in writers:

        users.append(w)
        num_samples.append(len(l))
        user_data[w] = {'x': [], 'y': []}

        size = 28, 28  # original image size is 128, 128
        for (f, c) in l:
            file_path = os.path.join(parent_path, f)
            img = Image.open(file_path)
            gray = img.convert('L')
            gray.thumbnail(size, Image.ANTIALIAS)
            arr = np.asarray(gray).copy()
            vec = arr.flatten()
            vec = vec / 255  # scale all pixel values to between 0 and 1
            vec = vec.tolist()

            nc = relabel_class(c)

            user_data[w]['x'].append(vec)
            user_data[w]['y'].append(nc)

        writer_count += 1
        all_writers += 1
        
        if writer_count == MAX_WRITERS or all_writers == len(writers):

            all_data = {}
            all_data['users'] = users
            all_data['num_samples'] = num_samples
            all_data['user_data'] = user_data

            file_name = 'all_data_%d.json' % json_index
            file_path = os.path.join(parent_path, 'data', 'FEMNIST', 'processed', file_name)

            print('writing %s' % file_name)

            with open(file_path, 'w') as outfile:
                json.dump(all_data, outfile)

            writer_count = 0
            json_index += 1
            
            users[:] = []
            num_samples[:] = []
            user_data.clear()

