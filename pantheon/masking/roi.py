#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to ROI manipulation."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Union, Optional, Literal, Any
import logging
import numpy as np
import nibabel as nib
import nilearn.masking as nlm

from ..image.cifti import decompose_dlabel
from ..utils.validation import parse_roi_id
from ..utils.typing import PathLike


def make_mask_from_index(
    data: np.ndarray, index_list: list[int], dtype: Literal["int", "bool"] = "int"
) -> np.ndarray:
    """Makes a binary mask array from a list of index values.

    Args:
        data: Input data array. Data type of this array should be int.
        index_list: A list of int. Each value in this list selects
            elements in data which match this value.
        dtype: Data type of the returned array. Default is int. If the
            returned array is used as a mask to index other arrays, it
            should be in bool type.

    Returns:
        A binary numpy array with same shape as data. Elements in
        data match any value in index_list are 1 in this array. And all
        other elements are 0.

    Raises:
        TypeError: data or index_list doesn't have the correct
            type.
        ValueError: dtype is not int or bool.
    """

    if not np.issubdtype(data.dtype, np.integer):
        raise TypeError(f"Argument data should have int data type.")
    if not (isinstance(index_list, list) and all(isinstance(i, int) for i in index_list)):
        raise TypeError("Argument index_list should be a list of int.")
    if dtype not in ["int", "bool"]:
        raise ValueError(f"Argument dtype is {dtype}. Valid: int, bool.")

    mask = np.zeros_like(data)
    for idx in index_list:
        mask += np.where(data == idx, 1, 0)
    mask = np.where(mask > 0, 1, 0).astype(dtype)
    return mask


def make_roi_from_spec(
    roi_id: str,
    roi_spec: dict[str, dict[str, Any]],
    atlas_file: list[Optional[PathLike]],
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """Makes ROI mask based on given ROI specification.

    Args:
        roi_id: ROI name.
        roi_spec: ROI specification. Usually it generates by the
            'read_roi_spec' function.
        atlas_file: Atlas files used for creating ROI. It should be a
            list of filenames corresponding to left and right brain
            hemisphere. Either file could be None. Both hemispheres
            could be the same file.

    Returns:
        A dict with 3 items. The keys are SurfaceL, SurfaceR and
        Volume, corresponding to the left, right brain hemisphere
        and the volume part. Usually a ROI could be either in
        surface or volume format, but not both.
        Surface mask is represented in a numpy array. Volume mask is
        represented in a nib.nifti1.Nifti1Image image.

    Raises:
        TypeError: atlas_file is not a list.
        ValueError: File of requested hemisphere is None in atlas_file.
        ValueError: ROIType field of the ROI specification is invalid.
    """

    if not isinstance(atlas_file, list):
        raise TypeError(
            "Argument atlas_file should be a list of two filenames "
            "correspoding to L and R hemispheres. (could be None or same file)"
        )
    roi_id, hemi = parse_roi_id(roi_id)
    if (hemi == "L" or hemi == "LR") and (atlas_file[0] is None):
        raise ValueError("Atlas file of left hemisphere is None.")
    if (hemi == "R" or hemi == "LR") and (atlas_file[1] is None):
        raise ValueError("Atlas file of right hemisphere is None.")

    spec = roi_spec[roi_id]
    roi_type = spec["ROIType"]
    roi_mask = {"Volume": None, "SurfaceL": None, "SurfaceR": None}
    # Volume
    if roi_type == "Volume":
        # load atlas data
        atlas_data = [None, None]
        for i in [0, 1]:
            if atlas_file[i]:
                # for mask image creation, assume both atlas images have same header
                atlas_img = nib.load(atlas_file[i])
                if not isinstance(atlas_img, nib.nifti1.Nifti1Image):
                    raise ValueError("For volume ROI, only support NIFTI atlas file.")
                atlas_data[i] = atlas_img.get_fdata().astype(np.int16)
        # make a mask from index
        if hemi == "L":
            mask = make_mask_from_index(atlas_data[0], spec["IndexL"])
        if hemi == "R":
            mask = make_mask_from_index(atlas_data[1], spec["IndexR"])
        if hemi == "LR":
            mask_lh = make_mask_from_index(atlas_data[0], spec["IndexL"])
            mask_rh = make_mask_from_index(atlas_data[1], spec["IndexR"])
            mask = mask_lh + mask_rh
        mask = np.where(mask > 0, 1, 0)
        mask_img = nib.Nifti1Image(mask, atlas_img.affine, atlas_img.header)
        roi_mask["Volume"] = mask_img
    # Surface (GIFTI, CIFTI)
    elif roi_type == "Surface":
        logging.disable(logging.CRITICAL)  # avoid CIFTI reading warning
        # load atlas data
        atlas_data = [None, None]
        for i, part in enumerate(["SurfaceL", "SurfaceR"]):
            if atlas_file[i]:
                atlas_img = nib.load(atlas_file[i])
                # GIFTI
                if isinstance(atlas_img, nib.gifti.gifti.GiftiImage):
                    atlas_data[i] = atlas_img.agg_data().astype(np.int16)
                # CIFTI
                elif isinstance(atlas_img, nib.cifti2.Cifti2Image):
                    atlas_data[i] = decompose_dlabel(atlas_img, dtype=np.int16)[part]
                else:
                    raise ValueError("For surface ROI, only support GIFTI or CIFTI atlas file.")
        logging.disable(logging.NOTSET)
        # make a mask from index
        # for CIFTI file, the vertex is in the 2nd dimension
        # use squeeze to make a 1d array as GIFTI file
        if hemi == "L":
            roi_mask["SurfaceL"] = make_mask_from_index(np.squeeze(atlas_data[0]), spec["IndexL"])
        if hemi == "R":
            roi_mask["SurfaceR"] = make_mask_from_index(np.squeeze(atlas_data[1]), spec["IndexR"])
        if hemi == "LR":
            roi_mask["SurfaceL"] = make_mask_from_index(np.squeeze(atlas_data[0]), spec["IndexL"])
            roi_mask["SurfaceR"] = make_mask_from_index(np.squeeze(atlas_data[1]), spec["IndexR"])
    else:
        raise ValueError(
            f"Invalid ROIType of {roi_id} in ROI specification. Valid: Volume, Surface."
        )

    return roi_mask


def unmask(
    data: np.ndarray,
    roi_mask: dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """Reshapes ROI data back into its original shape.

    Args:
        data: Any data (a numpy array) generates by the custom ROI data
            reading function in this package
            (e.g., read_preproc_func_cifti_roi).
        roi_mask: ROI mask dict used to read the data.

    Returns:
        A dict with 3 items contains the unmasked data. The keys are
        SurfaceL, SurfaceR and Volume, corresponding to the left, right
        brain hemisphere and the volume part.
        Surface mask is represented in a numpy array. Volume mask is
        represented in a nib.nifti1.Nifti1Image image.

    Raises:
        ValueError: data is not a 1-d or 2d numpy array.
        ValueError: roi_mask is invalid.
    """

    # Check data is a 1-d or 2-d data array
    if not (isinstance(data, np.ndarray) and data.ndim <= 2):
        raise ValueError("Argument data could only be a 1-d or 2-d numpy array.")
    # Check there's no surface part in roi_mask when it has volume part
    if (roi_mask["Volume"] is not None) and (
        (roi_mask["SurfaceL"] is not None) or (roi_mask["SurfaceR"] is not None)
    ):
        raise ValueError(
            "Argument roi_mask is invalid. "
            "It should be generated by the function 'mask_roi_from_spec'."
        )
    # Check there's at least one surface part when there's no surface part
    if (
        (roi_mask["Volume"] is None)
        and (roi_mask["SurfaceL"] is None)
        and (roi_mask["SurfaceR"] is None)
    ):
        raise ValueError(
            "Argument roi_mask is invalid. "
            "It should be generated by the function 'mask_roi_from_spec'."
        )

    data_um = {"SurfaceL": None, "SurfaceR": None, "Volume": None}
    if roi_mask["Volume"] is not None:
        data_um["Volume"] = nlm.unmask(data, roi_mask["Volume"])
    else:
        # Concatenate L and R surface mask if it's a bilateral ROI
        if (roi_mask["SurfaceL"] is not None) and (roi_mask["SurfaceR"] is not None):
            mask = np.hstack((roi_mask["SurfaceL"], roi_mask["SurfaceR"]))
        elif roi_mask["SurfaceL"] is not None:
            mask = roi_mask["SurfaceL"]
        elif roi_mask["SurfaceR"] is not None:
            mask = roi_mask["SurfaceR"]
        # Set data back to roi_mask shape using fancy indexing
        if data.ndim == 2:
            data_surf = np.zeros((data.shape[0], mask.shape[0]), dtype=data.dtype)
            data_surf[:, mask.astype("bool")] = data
        elif data.ndim == 1:
            data_surf = np.zeros((mask.shape[0]), dtype=data.dtype)
            data_surf[mask.astype("bool")] = data
        # Split unmasked L and R hemisphere data
        if (roi_mask["SurfaceL"] is not None) and (roi_mask["SurfaceR"] is not None):
            # assure lh data is always before rh data in a combined ROI
            n_vertex_lh = roi_mask["SurfaceL"].shape[0]
            if data.ndim == 2:
                data_um["SurfaceL"] = data_surf[:, :n_vertex_lh]
                data_um["SurfaceR"] = data_surf[:, n_vertex_lh:]
            if data.ndim == 1:
                data_um["SurfaceL"] = data_surf[:n_vertex_lh]
                data_um["SurfaceR"] = data_surf[n_vertex_lh:]
        elif roi_mask["SurfaceL"] is not None:
            data_um["SurfaceL"] = data_surf
        elif roi_mask["SurfaceR"] is not None:
            data_um["SurfaceR"] = data_surf

    return data_um
