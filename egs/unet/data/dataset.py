import sys
from os.path import dirname
sys.path.append(dirname(__file__))

import json
import os
import typing

import nibabel as nib
from torch.utils import data
import numpy as np

import torch
from torch.nn import functional as F
from torch.utils.data import Subset

import transformations
from torchvision import transforms as trfs


class NiftiFolder(data.Dataset):
    """
    A custom loader for .nii.gz files in a single folder.
    Each file should contain a scan of a single patient in one or more modalities.
    E.g.:
    scans/patient000.nii.gz
    scans/patient001.nii.gz
    scans/patient002.nii.gz
    where file scans/patient000.nii.gz contains scan of the patient 001 in 4 modalities:
    T1, T1gd, T2w, Flair
    (Note that the order of the modalities doesn't matter, however it should be consistent for whole dataset)
    """

    def __init__(self, paths: typing.List[str], transform: typing.Callable = None):
        self._files = paths
        self._transform = transform

    @classmethod
    def from_dir(cls, root: str, transforms: typing.Callable = None):
        files = [entry.path for entry in os.scandir(root)]
        return NiftiFolder(files, transforms)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> typing.Any:
        scan = nib.load(self._files[idx])
        scan_array = scan.get_fdata()

        if self._transform:
            scan_array = self._transform(scan_array)

        return scan_array


class CombinedDataset(data.Dataset):
    """
    Takes multiple datasets of the same length and combines them.
    On `__getitem__(n)` it returns a tuple containing nth element of each dataset.
    """

    def __init__(self, *datasets: data.Dataset, transform: transformations.CommonTransformation = None):
        assert all(len(dataset) == len(datasets[0]) for dataset in datasets), "Length of all datasets must be the same"
        self._datasets = datasets
        self._transform = transform

    def __len__(self) -> int:
        return len(self._datasets[0])

    def __getitem__(self, idx: int) -> typing.Tuple[typing.Any, ...]:
        if self._transform:
            return tuple(self._transform(*[dataset[idx] for dataset in self._datasets]))
        else:
            return tuple(dataset[idx] for dataset in self._datasets)


def read_dataset_json(path_to_json, key="training"):
    """
    Reads pairs of images and masks from json file.
    :param path_to_json: Path to the file from decathlon challange
    :return: Tuple with list of paths to images and list of path to masks
    """
    with open(path_to_json, "r") as json_file:
        json_dict = json.load(json_file)
    root = os.path.dirname(path_to_json)
    images_paths = [os.path.join(root, line["image"].replace("./", "")) for line in json_dict[key]]
    masks_paths = [os.path.join(root, line["label"].replace("./", "")) for line in json_dict[key]]
    return images_paths, masks_paths


class VolumesDataset(data.Dataset):
    def __init__(self, dataset_json, division_json, mode, input_size):
        volumes_transformations = trfs.Compose([transformations.NiftiToTorchDimensionsReorderTransformation(),
                                                trfs.Lambda(lambda x: torch.from_numpy(x)),
                                                trfs.Lambda(
                                                    lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[
                                                                                                  1] % 2 != 0 else x),
                                                transformations.StandardizeVolumeWithFilter(0),
                                                trfs.Lambda(lambda x: x.float())
                                                ])
        masks_transformations = trfs.Compose([trfs.Lambda(lambda x: np.expand_dims(x, 3)),
                                              transformations.NiftiToTorchDimensionsReorderTransformation(),
                                              trfs.Lambda(lambda x: torch.from_numpy(x)),
                                              transformations.OneHotEncoding([0, 1, 2, 3]),
                                              trfs.Lambda(
                                                  lambda x: F.pad(x, [0, 0, 0, 0, 5, 0]) if x.shape[
                                                                                                1] % 16 != 0 else x),
                                              trfs.Lambda(lambda x: x.float())
                                              ])
        common_transformations = transformations.ComposeCommon(
            [transformations.RandomCrop((input_size, input_size))])

        volumes_paths, masks_paths = read_dataset_json(dataset_json)
        volumes_set = NiftiFolder(volumes_paths, volumes_transformations)
        masks_set = NiftiFolder(masks_paths, masks_transformations)
        combined_set = CombinedDataset(volumes_set, masks_set, transform=common_transformations)
        with open(division_json, "r") as division_file:
            indices = json.load(division_file)

        assert mode in ["train", "valid", "test"]
        self.set = Subset(combined_set, indices[mode])

    def __getitem__(self, item):
        return self.set[item]

    def __len__(self) -> int:
        return len(self.set)
