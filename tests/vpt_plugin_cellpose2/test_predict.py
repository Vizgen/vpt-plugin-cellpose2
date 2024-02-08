from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pytest
from vpt_core.io.image import ImageSet

from tests.vpt_plugin_cellpose2 import TEST_DATA_ROOT
from vpt_plugin_cellpose2 import CellposeSegParameters, CellposeSegProperties
from vpt_plugin_cellpose2.predict import run


@dataclass(frozen=True)
class Circle:
    x: int
    y: int
    radius: int


def generate_images(image_size: int, cells: List[Circle]) -> Tuple[ImageSet, str, str]:
    dapi = np.ones((image_size, image_size), dtype=np.uint8)
    cellbound3 = np.ones((image_size, image_size), dtype=np.uint8)
    cellbound1 = np.ones((image_size, image_size), dtype=np.uint8)
    for cell in cells:
        cv2.circle(dapi, (cell.x, cell.y), int(cell.radius * 0.8), (255, 255, 255), -1)
        cv2.circle(cellbound3, (cell.x, cell.y), int(cell.radius * 1.5), (255, 255, 255), -1)
        cv2.circle(cellbound1, (cell.x, cell.y), int(cell.radius * 1.4), (255, 255, 255), -1)

    red, green, blue = "Cellbound1", "Cellbound3", "DAPI"
    images = ImageSet()
    images[red] = {i: cellbound1 for i in range(7)}
    images[green] = {i: cellbound3 for i in range(7)}
    images[blue] = {i: dapi for i in range(7)}

    return images


SEGMENTATION_PROPS = [
    {
        "model_dimensions": "2D",
        "channel_map": {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"},
        "model": "cyto2",
        "custom_weights": None,
    },
    {
        "model_dimensions": "2D",
        "channel_map": {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"},
        "model": None,
        "custom_weights": str(TEST_DATA_ROOT / "CP_20230830_093420"),
    },
    {
        "model_dimensions": "2D",
        "channel_map": {"red": None, "green": None, "blue": "DAPI"},
        "model": "cyto",
        "custom_weights": None,
    },
]
SEGMENTATION_PARAMS = [
    {"nuclear_channel": "DAPI", "entity_fill_channel": "all"},
    {"nuclear_channel": "blue", "entity_fill_channel": "grayscale"},
]


@pytest.mark.parametrize("seg_props", SEGMENTATION_PROPS)
@pytest.mark.parametrize("seg_params", SEGMENTATION_PARAMS)
def test_run_prediction(seg_props, seg_params) -> None:
    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images = generate_images(513, cells)

    model_dimensions = seg_props["model_dimensions"]
    channel_map = seg_props["channel_map"]
    model = seg_props["model"]
    custom_weights = seg_props["custom_weights"]

    nuclear_channel = seg_params["nuclear_channel"]
    entity_fill_channel = seg_params["entity_fill_channel"]

    properties = CellposeSegProperties(model_dimensions, channel_map, model, custom_weights)
    parameters = CellposeSegParameters(nuclear_channel, entity_fill_channel, 30, 0.95, 0.0, 0)
    mask = run(images, properties, parameters)
    for i in images.z_levels():
        labels = np.unique(mask[i, :, :])
        assert len(labels) > 0


def test_run_prediction_padding() -> None:
    cells = [Circle(20, 15, 10), Circle(30, 100, 10), Circle(100, 20, 15), Circle(210, 100, 15)]
    images = generate_images(256, cells)
    channel_map = {"red": "Cellbound1", "green": "Cellbound3", "blue": "DAPI"}
    nuclear_channel, entity_fill_channel = "DAPI", "grayscale"
    properties = CellposeSegProperties("2D", channel_map, None, str(TEST_DATA_ROOT / "CP_20230830_093420"))
    parameters = CellposeSegParameters(nuclear_channel, entity_fill_channel, 30, 0.95, 0.0, 0)
    mask = run(images, properties, parameters)
    for i in images.z_levels():
        labels = np.unique(mask[i, :, :])
        assert len(labels) > 0
