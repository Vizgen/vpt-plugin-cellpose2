from typing import Dict, Iterable, List, Optional, Union

import pandas as pd
from vpt_core.io.image import ImageSet
from vpt_core.segmentation.polygon_utils import generate_polygons_from_mask
from vpt_core.segmentation.seg_result import SegmentationResult
from vpt_core.segmentation.segmentation_base import SegmentationBase

from vpt_plugin_cellpose2 import CellposeSegParameters, CellposeSegProperties, predict


class SegmentationMethod(SegmentationBase):
    @staticmethod
    def run_segmentation(
        segmentation_properties: Dict,
        segmentation_parameters: Dict,
        polygon_parameters: Dict,
        result: List[str],
        images: Optional[ImageSet] = None,
        transcripts: Optional[pd.DataFrame] = None,
    ) -> Union[SegmentationResult, Iterable[SegmentationResult]]:
        properties = CellposeSegProperties(**segmentation_properties)
        parameters = CellposeSegParameters(**segmentation_parameters)

        masks = predict.run(images, properties, parameters)
        return generate_polygons_from_mask(masks, polygon_parameters)

    @staticmethod
    def validate_task(task: Dict) -> Dict:
        nuclear_channel = task["segmentation_parameters"].get("nuclear_channel", None)
        fill_channel = task["segmentation_parameters"].get("entity_fill_channel", None)
        channel_map = task["segmentation_properties"].get("channel_map", None)
        model = task["segmentation_properties"].get("model", None)
        custom_model = task["segmentation_properties"].get("custom_weights", None)

        channels = [input_data["image_channel"].lower() for input_data in task["task_input_data"]]
        if len(channels) == 0:
            raise ValueError("No channels have been specified in the task input data")
        channels.append("all")
        channels.append("grayscale")

        if not nuclear_channel:
            raise ValueError(f"Nuclear channel, {nuclear_channel}, is not specified")

        if not fill_channel:
            raise ValueError(f"Segment channel, {fill_channel}, is not specified")

        if nuclear_channel and nuclear_channel.lower() not in channels:
            raise ValueError(f"{nuclear_channel} is not in input channels")

        if fill_channel and fill_channel.lower() not in channels:
            raise ValueError(f"{fill_channel} is not in input channels")

        for color, chan in channel_map.items():
            if chan.lower() not in channels and chan.strip():
                raise ValueError(f"{chan} is not in input channels")

        if not model and not custom_model:
            raise ValueError("No model has been specified in the segmentation properties")

        return task
