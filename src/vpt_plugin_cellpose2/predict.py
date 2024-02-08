import warnings
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from cellpose import models
from cellpose.contrib import openvino_utils
from vpt_core.io.image import ImageSet

from vpt_plugin_cellpose2 import CellposeSegParameters, CellposeSegProperties

MINIMUM_IMAGE_SIZE_OPENCV = 513
warnings.filterwarnings("ignore", message=".*the `scipy.ndimage.filters` namespace is deprecated.*")


def run(images: ImageSet, properties: CellposeSegProperties, parameters: CellposeSegParameters) -> np.ndarray:
    """
    Runs Cellpose segmentation on an ImageSet using the specified properties and parameters. Returns a 3D label matrix mask
    the same where zero is background and each label is a segmentation object.
    """
    channels, index_map = assign_channel_index(properties.channel_map, parameters)
    image = convert_imageset_to_rgb_image(images, properties.channel_map)
    empty_z_levels, to_segment_z = detect_empty_z_planes(image, index_map)

    # If all z-levels are empty, there is nothing to do. Return an empty array.
    if len(empty_z_levels) == image.shape[0]:
        return np.zeros((image.shape[0],) + image.shape[1:-1])

    mask = extract_masks_with_cellpose(properties, parameters, image, to_segment_z, channels)

    # Fill in empty z-levels with zeros
    for z in empty_z_levels:
        mask = np.insert(mask, z, np.zeros(image.shape[1:-1]), axis=0)

    return mask


def extract_masks_with_cellpose(
    properties: CellposeSegProperties,
    parameters: CellposeSegParameters,
    image: np.ndarray,
    to_segment_z: List,
    channels: List,
) -> np.ndarray:
    """
    Runs Cellpose segmentation on a numpy 3D array using the specified properties and parameters. Returns a 3D label matrix
    mask the same where zero is background and each label is a segmentation object.
    """
    if properties.custom_weights:
        model = models.CellposeModel(gpu=False, pretrained_model=properties.custom_weights, net_avg=False)
    else:
        model = models.CellposeModel(gpu=False, model_type=properties.model, net_avg=False)

    model = openvino_utils.to_openvino(model)

    mask = model.eval(
        image[to_segment_z, ...],
        z_axis=0,
        channels=channels,
        channel_axis=len(image.shape) - 1,
        diameter=parameters.diameter,
        flow_threshold=parameters.flow_threshold,
        cellprob_threshold=parameters.cellprob_threshold,
        resample=False,
        min_size=parameters.minimum_mask_size,
        tile=True,
        do_3D=(properties.model_dimensions == "3D"),
    )[0]
    mask = mask.reshape((len(to_segment_z),) + image.shape[1:-1])
    return mask


def detect_empty_z_planes(image: np.ndarray, index_map: Dict) -> Tuple[Set, List]:
    """
    Based on the criteria that images that have cells in them have pixels of different intensity, labels z-planes of the images
    as "empty" if the standard deviation of the pixel intensity is less than 0.1. Cellpose is not run on "empty" images to save
    time.
    """
    empty_z_levels = set()
    channel_idx = [x - 1 for x in list(index_map.values()) if x > 0]

    for z_i, z_plane in enumerate(image):
        for channel_i in channel_idx:
            if z_plane[..., channel_i].std() < 0.1:
                empty_z_levels.add(z_i)
    to_segment_z = list(set(range(image.shape[0])).difference(empty_z_levels))
    return empty_z_levels, to_segment_z


def convert_imageset_to_rgb_image(images: ImageSet, channel_map: Dict) -> np.ndarray:
    """
    Converts ImageSet data into RGB image data using the user input from the Segmentation Properties "channel_map"
    """
    image_data = {"red": np.empty(0), "green": np.empty(0), "blue": np.empty(0)}
    image_shape: Tuple[int, ...] = (0, 0, 0)
    x_pad, y_pad = 0, 0

    # Iterate through each color and extract if from the ImageSet. If necessary, pad the image for OpenCV compatibility
    for image_color in image_data:
        if channel_map.get(image_color) and channel_map[image_color].strip():
            channel = channel_map[image_color].strip()
            image_data[image_color] = images.as_stack([channel])[..., 0]
            if any(x > 0 for x in image_data[image_color].shape):
                if any([dim < MINIMUM_IMAGE_SIZE_OPENCV for dim in image_data[image_color].shape[1:]]):
                    image_data[image_color], x_pad, y_pad = pad_image(image_data[image_color])
                image_shape = image_data[image_color].shape

    # If any colors were not present in the ImageSet, fill with zeros
    for image_color in image_data:
        if all(x == 0 for x in image_data[image_color].shape):
            image_data[image_color] = np.zeros(image_shape, dtype=np.uint8)

    image = cv2.merge(
        (
            image_data["red"],
            image_data["green"],
            image_data["blue"],
        )
    )

    # Remove the OpenCV compatibility pad if it was added
    if x_pad != 0 or y_pad != 0:
        image = image[:, 0 : MINIMUM_IMAGE_SIZE_OPENCV - x_pad, 0 : MINIMUM_IMAGE_SIZE_OPENCV - y_pad, :]

    assert any([dim > 0 for dim in image_shape]), "Image size is (0,0,0)"
    return image


def assign_channel_index(channel_map: Dict, parameters: CellposeSegParameters) -> Tuple[List, Dict]:
    """
    Map from user input strings specifying the images to use for segmentation (which may be colors like "green" or image
    channels like "DAPI") to the index value of that channel in the np.ndarray input to cellpose
    """
    index_map = {}
    for i, chan in enumerate(list(channel_map.values())):
        i += 1
        if chan and chan.strip():
            index_map[chan.lower()] = i
    index_map["grayscale"] = 0

    index_map_color = {}
    for color in list(channel_map.keys()):
        if channel_map[color]:
            index_map_color[color] = channel_map[color].lower()
    index_map_color["all"] = "grayscale"

    channels = []
    for user_channel in [parameters.entity_fill_channel.lower(), parameters.nuclear_channel.lower()]:
        if user_channel in index_map_color.keys():
            channels.append(index_map[index_map_color.get(user_channel)])
        elif user_channel in index_map_color.values():
            channels.append(index_map[user_channel])
        else:
            raise ValueError
    return channels, index_map


def pad_image(image_array: np.ndarray) -> Tuple[np.ndarray, int, int]:
    """
    OpenCV crashes when creating RGB images from very small inputs. This fuction pads the images to be large enough to process
    """
    x_pad = MINIMUM_IMAGE_SIZE_OPENCV - image_array.shape[1]
    y_pad = MINIMUM_IMAGE_SIZE_OPENCV - image_array.shape[2]
    image_array = np.pad(image_array, [(0, 0), (0, x_pad), (0, y_pad)], mode="constant", constant_values=0)
    return image_array, x_pad, y_pad
