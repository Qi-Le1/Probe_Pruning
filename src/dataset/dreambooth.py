import numpy as np
import os
import pickle
import torch
import requests
from config import cfg
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset

from module import check_exists, makedir_exist_ok, save, load
from .utils import download_url, extract_file, make_classes_counts


class DreamBooth(Dataset):
    data_name = 'DreamBooth'
    api_url = f'https://api.github.com/repos/google/dreambooth/contents/dataset'

    def __init__(
            self,
            root,
            split,
            model,
            tokenizer,
            instance_data_dir,
            instance_prompt,
            class_data_dir=None,
            class_prompt=None,
            transform=None
    ):
        self.root = os.path.expanduser(root)
        self.split = split
        self.tokenizer = tokenizer
        self.instance_data_dir = os.path.join(self.processed_folder, instance_data_dir)
        self.instance_prompt = instance_prompt
        self.class_data_dir = os.path.join(self.processed_folder, class_data_dir)
        self.class_prompt = class_prompt
        self.transform = transform

        if not check_exists(self.processed_folder):
            self.download_github_directory(self.api_url, self.processed_folder)
        if not check_exists(self.class_data_dir):
            self.process(model)
        self.make_data()
        return

    def __getitem__(self, index):
        input = {}
        instance_image = Image.open(self.instance_images_path[index % self.num_instance_images])
        if not instance_image.mode == "RGB":
            instance_image = instance_image.convert("RGB")
        input["instance_images"] = self.transform(instance_image)
        input["instance_prompt_ids"] = self.tokenizer(
            self.instance_prompt,
            truncation=True,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids

        if self.class_data_dir:
            class_image = Image.open(self.class_images_path[index % self.num_class_images])
            if not class_image.mode == "RGB":
                class_image = class_image.convert("RGB")
            input["class_images"] = self.transform(class_image)
            input["class_prompt_ids"] = self.tokenizer(
                self.class_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            ).input_ids
        return input

    def __len__(self):
        return self._length

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'processed')

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'raw')

    def generate_prior_data(self, model):
        model_name = cfg['model_name']
        model.to(cfg['device'])
        for i in range(cfg[cfg['model_name']]['num_class_image']):
            prompt = self.class_prompt
            image = model(prompt, num_inference_steps=cfg[model_name]['num_inference_steps'],
                          guidance_scale=cfg[model_name]['guidance_scale']).images[0]
            image_path = os.path.join(self.class_data_dir, f"class_pic_{i}.png")
            # Save the image to the specified path
            image.save(image_path)
        return

    def process(self, model):
        makedir_exist_ok(self.class_data_dir)
        self.generate_prior_data(model)
        return

    def __repr__(self):
        fmt_str = 'Dataset {}\nSize: {}\nRoot: {}\nSplit: {}\nTransforms: {}'.format(
            self.__class__.__name__, self.__len__(), self.root, self.split, self.transform.__repr__())
        return fmt_str

    def make_data(self):
        self.instance_images_path = list(Path(self.instance_data_dir).iterdir())
        self.num_instance_images = len(self.instance_images_path)
        self._length = self.num_instance_images

        if self.class_data_dir is not None:
            self.class_images_path = list(Path(self.class_data_dir).iterdir())
            self.num_class_images = len(self.class_images_path)
            self._length = max(self.num_class_images, self.num_instance_images)
        else:
            self.class_data_root = None
        return None

    def download_github_directory(self, api_url, destination):
        # Make a request to get the contents of the directory
        contents_response = requests.get(api_url)
        contents_response.raise_for_status()  # Raise an error if the request failed

        # Iterate over the files and directories in the current directory
        print('download dreambooth dataset start')
        for file_info in contents_response.json():
            if file_info['type'] == 'file':
                # Download each file
                file_response = requests.get(file_info['download_url'])
                file_response.raise_for_status()

                # Write the file to the local file system
                file_path = os.path.join(destination, file_info['name'])
                with open(file_path, 'wb') as f:
                    f.write(file_response.content)
                # print(f'Downloaded {file_info["name"]}')
            elif file_info['type'] == 'dir':
                # Create a new directory locally for the subdirectory
                new_destination = os.path.join(destination, file_info['name'])
                makedir_exist_ok(new_destination)

                # Recursively call this function for the new directory
                new_api_url = file_info['url']  # URL for the subdirectory
                self.download_github_directory(new_api_url, new_destination)
        print('download dreambooth dataset end')
        return
