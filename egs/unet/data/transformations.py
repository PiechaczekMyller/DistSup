import abc
import random
import typing
import utils
from functools import singledispatch
import numpy as np
import torch
from torch.nn import functional as F

CHANNELS_IDX = 3


class HistogramMatchingTransformation:
    """
    Transformation performing histogram matching of volumes
    """

    def __init__(self, template: np.ndarray):
        """

        Args:
            template (np.ndarray): Template which will be used for histogram
                matching in future calls.
                Dimensions - (Height, Width, Depth, Channel)
        """
        self.template = template

    def __call__(self, volume: np.ndarray) -> np.ndarray:
        """
        Args:
            volume (np.ndarray): Volume to be matched.
                Dimensions - (Height, Width, Depth, Channel)

        Returns:
             np.ndarray: Normalized volume with the same shape as input volume
        """
        assert self.template.ndim == volume.ndim
        result = np.zeros_like(volume)
        for channel in range(volume.shape[CHANNELS_IDX]):
            result[..., channel] = self.hist_match(volume[..., channel],
                                                   self.template[..., channel])
        return result

    @staticmethod
    def hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
        """
            Adjust the pixel values of an image such that its histogram
            matches that of a target image

            Args:
                source (np.ndarray): Image to transform; the histogram is
                    computed over the flattened array
                template (np.ndarray): Template image; can have different
                    dimensions to source
            Returns:
                np.ndarray: The transformed output image
                    of the same shape as input
            """
        positive = source > 0
        result = np.zeros_like(source)
        source = source[positive].ravel()
        template = template[template > 0].ravel()
        # get the set of unique pixel values and their corresponding indices and
        # counts
        source_values, bin_idx, source_counts = np.unique(source,
                                                          return_inverse=True,
                                                          return_counts=True)
        template_values, template_counts = np.unique(template,
                                                     return_counts=True)
        # take the cumsum of the counts and normalize by the number of pixels to
        # get the empirical cumulative distribution functions for the source and
        # template images (maps pixel value --> quantile)
        source_quantiles = np.cumsum(source_counts).astype(np.float64)
        source_quantiles /= source_quantiles[-1]
        template_quantiles = np.cumsum(template_counts).astype(np.float64)
        template_quantiles /= template_quantiles[-1]

        # interpolate linearly to find the pixel values in the template image
        # that correspond most closely to the quantiles in the source image
        interp_t_values = np.interp(source_quantiles, template_quantiles,
                                    template_values)
        result[positive] = interp_t_values[bin_idx]
        return result


@singledispatch
def reorder(img: typing.Any):
    raise TypeError("Img should be either np.ndarray of torch.Tensor")


H = 0
W = 1
D = 2
C = 3


@reorder.register(np.ndarray)
def _(img: np.ndarray) -> np.ndarray:
    transformed = np.moveaxis(img, [H, W, D, C], [3, 2, 1, 0])
    return transformed


@reorder.register(torch.Tensor)
def _(img: torch.Tensor) -> torch.Tensor:
    transformed = img.permute(C, D, W, H)
    return transformed


class NiftiToTorchDimensionsReorderTransformation:
    """
    Changes dimensions order from nifti (H,W,D,C) to torch convention (C,D,H,W).
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        assert len(img.shape) == 4, "Tensor should have 4 dimensions (H,W,D,C)"

        transformed = reorder(img)
        return transformed


@singledispatch
def add_channel_dim(img: typing.Any):
    raise TypeError("Img should be either np.ndarray of torch.Tensor")


@add_channel_dim.register(np.ndarray)
def _(img: np.ndarray) -> np.ndarray:
    transformed = np.expand_dims(img, 3)
    return transformed


@add_channel_dim.register(torch.Tensor)
def _(img: torch.Tensor) -> torch.Tensor:
    transformed = torch.unsqueeze(img, 0)
    return transformed


class AddChannelDimToMaskTransformation:
    """
    Adds additional channel dimension
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        assert img.ndim == 3, "Img should have 3 dimensions (D,H,W)"
        transformed = add_channel_dim(img)
        return transformed


class BinarizationTransformation:
    """
    Changes image to binary mask, all values above 1 would be changed to 1.
    """

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor]) -> typing.Union[np.ndarray, torch.Tensor]:
        transformed = utils.copy(img)
        transformed[transformed > 0] = 1
        return transformed


class ResizeVolumeTransformation:
    """
    Resizes the volume spatial dimensions (H,W) to the given size.
    For details, see nn.functional.interpolate.
    """

    def __init__(self, size: typing.Union[int, typing.Tuple[int, int]]):
        self.size = size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        assert tensor.ndimension == 4, "Tensor should have 4 dimensions (C,D,H,W)"

        out = F.interpolate(tensor, size=self.size)  # The resize operation on tensor.
        return out


class StandardizeVolume:
    """
    Performs standardization on the volume
    (mean -> 0, std -> 1 with standard metrics)
    Standardization is done on the volumetric dimensions,
    if multiple channels are given, standardization is done channel-wise.
    """

    def __init__(self, epsilon: float = 1e-6):
        self.epsilon = epsilon

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        means = tensor.mean(dim=(1, 2, 3), keepdims=True)
        stds = tensor.std(dim=(1, 2, 3), keepdims=True)
        transformed = (tensor - means) / (stds + self.epsilon)
        return transformed


class StandardizeVolumeWithFilter:
    """
    Performs standardization with filtering on the volume
    (mean -> 0, std -> 1 with standard metrics)
    Standardization is done on the volumetric dimensions,
    if multiple channels are given, standardization is done channel-wise.
    """

    def __init__(self, value_to_filter: float, epsilon: float = 1e-6):
        self.value_to_filter = value_to_filter
        self.epsilon = epsilon

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        means = torch.zeros(tensor.shape[0], 1, 1, 1).to(tensor.dtype)
        stds = torch.zeros(tensor.shape[0], 1, 1, 1).to(tensor.dtype)
        for channel_id in range(tensor.shape[0]):
            filtered = self._filter(tensor[channel_id])
            means[channel_id, ...] = filtered.mean()
            stds[channel_id, ...] = filtered.std()
        transformed = (tensor - means) / (stds + self.epsilon)
        return transformed

    def _filter(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor[tensor != self.value_to_filter]


class OneHotEncoding:
    """
    From multiclass label image with one channel it creates N channel image,
    where each channel is a mask of different class.
    """

    def __init__(self, classes: typing.List[int]):
        self.classes = classes

    def __call__(self, img: typing.Union[np.ndarray, torch.Tensor], ) -> typing.Union[np.ndarray, torch.Tensor]:
        assert img.ndimension() == 4, "Tensor should have 4 dimensions (C,D,H,W)"
        transformed = one_hot_encoding(img, self.classes)
        return transformed


@singledispatch
def one_hot_encoding(mask: typing.Any, classes: typing.List[int]):
    """
        Function that transforms 1 channel mask to mask with N channels.
        :param mask: mask to transform
        :param classes: list of labels present in dataset. It is needed as some instances may not contain all classes.
        """
    raise TypeError("Mask should be either np.ndarray of torch.Tensor")


@one_hot_encoding.register(np.ndarray)
def _(mask: np.ndarray, classes: typing.List[int]) -> np.ndarray:
    new_shape = [len(classes)] + list(mask.shape[1:])
    transformed = np.zeros(new_shape)
    for class_id, label in enumerate(classes):
        transformed[class_id][mask[0, ...] == label] = 1
    return transformed


@one_hot_encoding.register(torch.Tensor)
def _(mask: torch.Tensor, classes: typing.List[int]) -> torch.Tensor:
    new_shape = [len(classes)] + list(mask.shape[1:])
    transformed = torch.zeros(new_shape)

    for class_id, label in enumerate(classes):
        transformed[class_id][mask[0, ...] == label] = 1
    return transformed


class CommonTransformation(abc.ABC):
    """
    Interface for transformations that are supposed to transform multiple inputs in the same way.
    """

    @abc.abstractmethod
    def __call__(self, imgs: typing.List[typing.Union[np.ndarray, torch.Tensor]]):
        raise NotImplementedError


class RandomCrop(CommonTransformation):
    """
    Extracts random patch from volume of the given size.
    :param size: Tuple containing sizes (H,W) of the desired patch.
    """

    def __init__(self, size: typing.Tuple[int, int]):
        self.size = size

    def __call__(self, *imgs: typing.Union[np.ndarray, torch.Tensor]) -> typing.Tuple[
        typing.Union[np.ndarray, torch.Tensor],
        typing.Union[np.ndarray, torch.Tensor]]:
        assert all(
            img[0, 0, ...].shape == imgs[0][0, 0, ...].shape for img in imgs), "W and H of all images must be the same"

        max_x = imgs[0].shape[2] - self.size[0]
        max_y = imgs[0].shape[3] - self.size[1]
        x, y = random.randint(0, max_x), random.randint(0, max_y)
        transformed = []
        for img in imgs:
            transformed_img = utils.copy(img)
            transformed_img = transformed_img[:, :, x:x + self.size[0], y:y + self.size[1]]
            transformed.append(transformed_img)

        return tuple(transformed)


class ComposeCommon(CommonTransformation):
    """
    Compose multiple transformations into one
    :param transform: List of transforms to perform.
    """

    def __init__(self, transforms: typing.List[CommonTransformation]):
        self.transforms = transforms

    def __call__(self, *imgs: typing.Union[np.ndarray, torch.Tensor]) -> typing.Tuple[
        typing.Union[np.ndarray, torch.Tensor],
        typing.Union[np.ndarray, torch.Tensor]]:
        assert all(
            img[0, 0, ...].shape == imgs[0][0, 0, ...].shape for img in imgs), "W and H of all images must be the same"
        transformed = [utils.copy(img) for img in imgs]
        for transform in self.transforms:
            transformed = transform(*transformed)
        return transformed
