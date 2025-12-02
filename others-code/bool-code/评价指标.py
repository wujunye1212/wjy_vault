"""
Extended COCO evaluator with additional size-based metrics: APvtiny and APtiny.
"""
import os
import contextlib
import copy
import numpy as np
import torch
from typing import List

from faster_coco_eval import COCO, COCOeval_faster
import faster_coco_eval.core.mask as mask_util
from ....core import register
from ....misc import dist_utils
from ..coco_eval import CocoEvaluator


class ExtendedParams(object):
    """Extended parameters for COCO evaluation with additional area ranges."""

    def setDetParams(self):
        """Set parameters for detection evaluation with extended area ranges."""
        self.maxDets = [1, 10, 100]
        self.areaRng = [
            [0 ** 2, 1e5 ** 2],  # all
            [0 ** 2, 8 ** 2],  # vtiny (very tiny) - new
            [8 ** 2, 16 ** 2],  # tiny - new
            [0 ** 2, 32 ** 2],  # small
            [32 ** 2, 96 ** 2],  # medium
            [96 ** 2, 1e5 ** 2],  # large
        ]
        self.areaRngLbl = ["all", "vtiny", "tiny", "small", "medium", "large"]


@register()
class ExtendedCocoEvaluator(CocoEvaluator):
    """Extended COCO evaluator with additional metrics for very tiny and tiny objects."""

    def __init__(self, coco_gt, iou_types):
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt: COCO = coco_gt
        self.iou_types = iou_types

        self.coco_eval = {}
        for iou_type in iou_types:
            # Create a custom evaluator with extended parameters
            coco_eval = COCOeval_faster(coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
            # Replace the default parameters with our extended parameters
            extended_params = ExtendedParams()
            extended_params.setDetParams()

            # Copy over the original parameters that we're not modifying
            extended_params.imgIds = coco_eval.params.imgIds
            extended_params.catIds = coco_eval.params.catIds
            extended_params.iouThrs = coco_eval.params.iouThrs
            extended_params.recThrs = coco_eval.params.recThrs
            extended_params.useCats = coco_eval.params.useCats
            extended_params.compute_rle = coco_eval.params.compute_rle
            extended_params.compute_boundary = coco_eval.params.compute_boundary
            extended_params.iouType = coco_eval.params.iouType
            extended_params.imgCountLbl = coco_eval.params.imgCountLbl

            # Set the extended parameters
            coco_eval.params = extended_params

            # Initialize the evaluation metrics array sizes to match our extended parameters
            if not coco_eval.lvis_style:
                coco_eval.eval = {
                    'params': extended_params,
                    'counts': [len(extended_params.imgIds), len(extended_params.catIds), len(extended_params.areaRng),
                               len(extended_params.maxDets), len(extended_params.iouThrs)],
                    'precision': np.zeros(
                        [len(extended_params.iouThrs), len(extended_params.recThrs), len(extended_params.catIds),
                         len(extended_params.areaRng), len(extended_params.maxDets)]),
                    'recall': np.zeros(
                        [len(extended_params.iouThrs), len(extended_params.catIds), len(extended_params.areaRng),
                         len(extended_params.maxDets)]),
                }

            self.coco_eval[iou_type] = coco_eval

        self.img_ids = []
        self.eval_imgs = {k: [] for k in iou_types}

    def cleanup(self):
        """Override cleanup to initialize with extended parameters."""
        self.coco_eval = {}
        for iou_type in self.iou_types:
            coco_eval = COCOeval_faster(self.coco_gt, iouType=iou_type, print_function=print, separate_eval=True)
            # Replace the default parameters with our extended parameters
            extended_params = ExtendedParams()
            extended_params.setDetParams()

            # Copy over the original parameters that we're not modifying
            extended_params.imgIds = coco_eval.params.imgIds
            extended_params.catIds = coco_eval.params.catIds
            extended_params.iouThrs = coco_eval.params.iouThrs
            extended_params.recThrs = coco_eval.params.recThrs
            extended_params.useCats = coco_eval.params.useCats
            extended_params.compute_rle = coco_eval.params.compute_rle
            extended_params.compute_boundary = coco_eval.params.compute_boundary
            extended_params.iouType = coco_eval.params.iouType
            extended_params.imgCountLbl = coco_eval.params.imgCountLbl

            # Set the extended parameters
            coco_eval.params = extended_params

            self.coco_eval[iou_type] = coco_eval

        self.img_ids = []
        self.eval_imgs = {k: [] for k in self.iou_types}

    def update(self, predictions):
        """Override update to handle the extended area ranges."""
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            results = self.prepare(predictions, iou_type)
            coco_eval = self.coco_eval[iou_type]

            # suppress pycocotools prints
            with open(os.devnull, 'w') as devnull:
                with contextlib.redirect_stdout(devnull):
                    coco_dt = self.coco_gt.loadRes(results) if results else COCO()
                    coco_eval.cocoDt = coco_dt
                    coco_eval.params.imgIds = list(img_ids)

                    # Ensure we're using our extended parameters
                    if not hasattr(coco_eval.params, 'areaRngLbl') or 'vtiny' not in coco_eval.params.areaRngLbl:
                        extended_params = ExtendedParams()
                        extended_params.setDetParams()

                        # Copy over the parameters we're not modifying
                        extended_params.imgIds = coco_eval.params.imgIds
                        extended_params.catIds = coco_eval.params.catIds
                        extended_params.iouThrs = coco_eval.params.iouThrs
                        extended_params.recThrs = coco_eval.params.recThrs
                        extended_params.useCats = coco_eval.params.useCats
                        extended_params.compute_rle = coco_eval.params.compute_rle
                        extended_params.compute_boundary = coco_eval.params.compute_boundary
                        extended_params.iouType = coco_eval.params.iouType
                        extended_params.imgCountLbl = coco_eval.params.imgCountLbl

                        # Set the extended parameters
                        coco_eval.params = extended_params

                    try:
                        coco_eval.evaluate()
                    except Exception as e:
                        print(f"Error during evaluation: {e}")
                        continue

            # Instead of trying to manipulate the _evalImgs_cpp directly,
            # we'll just append the evaluation results to self.eval_imgs
            # and handle the reshaping later in synchronize_between_processes
            try:
                # Check if _evalImgs_cpp exists and is not None
                if hasattr(coco_eval, '_evalImgs_cpp') and coco_eval._evalImgs_cpp is not None:
                    # For faster_coco_eval, _evalImgs_cpp might be a special object
                    # We'll just store it as is and handle it in synchronize_between_processes
                    self.eval_imgs[iou_type].append(coco_eval._evalImgs_cpp)
                else:
                    # If _evalImgs_cpp doesn't exist or is None, we'll create a placeholder
                    print(f"Warning: _evalImgs_cpp is None for {iou_type}")
                    # Create a placeholder of the right shape
                    expected_shape = (len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(img_ids))
                    self.eval_imgs[iou_type].append(np.zeros(expected_shape))
            except Exception as e:
                print(f"Error appending evaluation results: {e}")
                # If there's an error, we'll just append a placeholder
                expected_shape = (len(coco_eval.params.catIds), len(coco_eval.params.areaRng), len(img_ids))
                self.eval_imgs[iou_type].append(np.zeros(expected_shape))

    def synchronize_between_processes(self):
        """Override synchronize_between_processes to handle the extended area ranges."""
        for iou_type in self.iou_types:
            # Custom implementation to handle different types of eval_imgs
            all_img_ids = dist_utils.all_gather(self.img_ids)
            all_eval_imgs = dist_utils.all_gather(self.eval_imgs[iou_type])

            # Merge image IDs
            merged_img_ids = []
            for p in all_img_ids:
                merged_img_ids.extend(p)

            # Convert to numpy array and get unique IDs
            merged_img_ids = np.array(merged_img_ids)
            merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
            merged_img_ids = merged_img_ids.tolist()

            # Handle the eval_imgs differently based on their type
            coco_eval = self.coco_eval[iou_type]

            # Set the image IDs
            coco_eval.params.imgIds = merged_img_ids
            coco_eval._paramsEval = copy.deepcopy(coco_eval.params)

            # Handle the case where eval_imgs is empty
            if not all_eval_imgs or not any(all_eval_imgs):
                # Create an empty array of the right shape
                coco_eval._evalImgs_cpp = []
                continue

            # Try to handle ImageEvaluation objects or numpy arrays
            try:
                # First, check if we're dealing with ImageEvaluation objects
                if hasattr(all_eval_imgs[0][0], '__class__') and 'ImageEvaluation' in str(
                        all_eval_imgs[0][0].__class__):
                    # We're dealing with ImageEvaluation objects, just flatten the list
                    merged_eval_imgs = []
                    for p in all_eval_imgs:
                        merged_eval_imgs.extend(p)
                    coco_eval._evalImgs_cpp = merged_eval_imgs
                else:
                    # We're dealing with numpy arrays, try to concatenate them
                    try:
                        # Try to reshape arrays if needed
                        reshaped_arrays = []
                        for eval_img_list in all_eval_imgs:
                            for eval_img in eval_img_list:
                                # Check if it's already a numpy array
                                if isinstance(eval_img, np.ndarray):
                                    # If it's 1D, try to reshape it
                                    if eval_img.ndim == 1:
                                        # Calculate the expected shape
                                        cat_count = len(coco_eval.params.catIds)
                                        area_count = len(coco_eval.params.areaRng)
                                        img_count = len(eval_img) // (cat_count * area_count)

                                        # Reshape to 3D
                                        eval_img = eval_img.reshape(cat_count, area_count, img_count)

                                    reshaped_arrays.append(eval_img)
                                else:
                                    # If it's not a numpy array, convert it
                                    reshaped_arrays.append(np.array(eval_img))

                        # Now try to concatenate along axis 2 (imgIds)
                        if reshaped_arrays:
                            merged_eval_imgs = np.concatenate(reshaped_arrays, axis=2).ravel().tolist()
                        else:
                            merged_eval_imgs = []
                    except Exception as e:
                        print(f"Error concatenating eval_imgs: {e}")
                        # If concatenation fails, just flatten the arrays
                        merged_eval_imgs = []
                        for p in all_eval_imgs:
                            for eval_img in p:
                                if isinstance(eval_img, np.ndarray):
                                    merged_eval_imgs.extend(eval_img.ravel().tolist())
                                else:
                                    merged_eval_imgs.extend(eval_img)

                    coco_eval._evalImgs_cpp = merged_eval_imgs
            except Exception as e:
                print(f"Error processing eval_imgs: {e}")
                # If all else fails, create an empty array
                coco_eval._evalImgs_cpp = []

    def accumulate(self):
        """Override accumulate to handle the extended area ranges."""
        for iou_type, coco_eval in self.coco_eval.items():
            print(f"Accumulating evaluation results for {iou_type}...")

            # Call the original accumulate method
            try:
                coco_eval.accumulate()
            except Exception as e:
                print(f"Error during accumulation: {e}")
                # If there's an error, we need a different approach
                # Let's try to modify the standard COCO evaluator to use our extended parameters

                # Create a simple version of the evaluation data
                # This is a fallback mechanism if the normal accumulation fails
                print("Using fallback accumulation mechanism...")

                # Initialize evaluation arrays
                p = coco_eval.params
                coco_eval.eval = {
                    'params': p,
                    'counts': [len(p.imgIds), len(p.catIds), len(p.areaRng), len(p.maxDets), len(p.iouThrs)],
                    'precision': np.zeros(
                        [len(p.iouThrs), len(p.recThrs), len(p.catIds), len(p.areaRng), len(p.maxDets)]),
                    'recall': np.zeros([len(p.iouThrs), len(p.catIds), len(p.areaRng), len(p.maxDets)]),
                }

                # We'll use placeholder values for the metrics we can't compute
                # This will allow the code to continue running, but the metrics for vtiny and tiny
                # will not be accurate in this fallback mode
                print("Warning: Using placeholder values for extended metrics.")

    def summarize(self):
        """Override summarize to handle the extended metrics."""
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))

            # Define a custom summarize function to include the new metrics
            def custom_summarize():
                """Compute and display summary metrics for evaluation results."""
                _count = 19 if coco_eval.lvis_style else 16  # Increased by 2 for new metrics
                stats = np.zeros((_count,))

                try:
                    stats[0] = coco_eval._summarize(1, maxDets=coco_eval.params.maxDets[-1])  # AP_all
                    stats[1] = coco_eval._summarize(1, iouThr=0.5, maxDets=coco_eval.params.maxDets[-1])  # AP_50
                    stats[2] = coco_eval._summarize(1, iouThr=0.75, maxDets=coco_eval.params.maxDets[-1])  # AP_75

                    # Add the new metrics - use string names directly
                    # We need to handle potential errors here if the vtiny/tiny metrics can't be computed
                    try:
                        stats[3] = coco_eval._summarize(1, areaRng="vtiny",
                                                        maxDets=coco_eval.params.maxDets[-1])  # AP_vtiny
                    except ValueError:
                        print("Warning: Could not compute AP_vtiny, using 0.0")
                        stats[3] = 0.0

                    try:
                        stats[4] = coco_eval._summarize(1, areaRng="tiny",
                                                        maxDets=coco_eval.params.maxDets[-1])  # AP_tiny
                    except ValueError:
                        print("Warning: Could not compute AP_tiny, using 0.0")
                        stats[4] = 0.0

                    # Existing metrics
                    stats[5] = coco_eval._summarize(1, areaRng="small",
                                                    maxDets=coco_eval.params.maxDets[-1])  # AP_small
                    stats[6] = coco_eval._summarize(1, areaRng="medium",
                                                    maxDets=coco_eval.params.maxDets[-1])  # AP_medium
                    stats[7] = coco_eval._summarize(1, areaRng="large",
                                                    maxDets=coco_eval.params.maxDets[-1])  # AP_large

                    if coco_eval.lvis_style:
                        stats[16] = coco_eval._summarize(1, maxDets=coco_eval.params.maxDets[-1],
                                                         freq_group_idx=0)  # APr
                        stats[17] = coco_eval._summarize(1, maxDets=coco_eval.params.maxDets[-1],
                                                         freq_group_idx=1)  # APc
                        stats[18] = coco_eval._summarize(1, maxDets=coco_eval.params.maxDets[-1],
                                                         freq_group_idx=2)  # APf

                    stats[8] = coco_eval._summarize(0, maxDets=coco_eval.params.maxDets[0])  # AR_first or AR_all
                    if len(coco_eval.params.maxDets) >= 2:
                        stats[9] = coco_eval._summarize(0, maxDets=coco_eval.params.maxDets[1])  # AR_second
                    if len(coco_eval.params.maxDets) >= 3:
                        stats[10] = coco_eval._summarize(0, maxDets=coco_eval.params.maxDets[2])  # AR_third

                    # Add corresponding AR metrics
                    try:
                        stats[11] = coco_eval._summarize(0, areaRng="vtiny",
                                                         maxDets=coco_eval.params.maxDets[-1])  # AR_vtiny
                    except ValueError:
                        print("Warning: Could not compute AR_vtiny, using 0.0")
                        stats[11] = 0.0

                    try:
                        stats[12] = coco_eval._summarize(0, areaRng="tiny",
                                                         maxDets=coco_eval.params.maxDets[-1])  # AR_tiny
                    except ValueError:
                        print("Warning: Could not compute AR_tiny, using 0.0")
                        stats[12] = 0.0

                    stats[13] = coco_eval._summarize(0, areaRng="small",
                                                     maxDets=coco_eval.params.maxDets[-1])  # AR_small
                    stats[14] = coco_eval._summarize(0, areaRng="medium",
                                                     maxDets=coco_eval.params.maxDets[-1])  # AR_medium
                    stats[15] = coco_eval._summarize(0, areaRng="large",
                                                     maxDets=coco_eval.params.maxDets[-1])  # AR_large
                except Exception as e:
                    print(f"Error during summarization: {e}")
                    # If there's an error, we'll return zeros for all metrics
                    print("Warning: Using zeros for all metrics due to error.")

                return stats

            # Replace the original stats with our extended stats
            coco_eval.stats = custom_summarize()

            # Print the summarized results (this will print all metrics including our new ones)
            try:
                coco_eval.summarize()
            except Exception as e:
                print(f"Error during summarization display: {e}")
                print("AP, AR metrics:")
                for i, stat in enumerate(coco_eval.stats):
                    print(f"Metric {i}: {stat:.3f}")