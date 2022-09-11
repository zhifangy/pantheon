#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to CIFTI file manipulation."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Union, Optional, Literal, Any
from pathlib import Path
import tempfile
import numpy as np
import nibabel as nib
import nilearn.image as nli
import nilearn.masking as nlm
from scipy.stats import zscore

from .gifti import make_gifti_image, make_gifti_label_image
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


################################
# Manipulate CIFTI python object
################################


def decompose_dscalar(
    img: nib.cifti2.Cifti2Image, dtype: Any = np.float32
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """
    Splits a CIFTI dscalar image into surface and volume parts.

    Note: By default, CIFTI file represents samples in columns.

    Args:
        img: A nib.cifti2.Cifti2Image object.
        dtype: Data type of the splitted data.

    Returns:
        A dict contains splitted CIFTI data. The keys are SurfaceL,
        SurfaceR, and Volume.
    """

    data = {"SurfaceL": None, "SurfaceR": None, "Volume": None}
    # Read data in CIFTI file
    data_cifti = img.get_fdata(dtype=np.float64).astype(dtype)
    # Get BrainModel axis (assume the last dim for dscalar or dtseries)
    axis_bm = img.header.get_axis(1)
    # Split data for each brain structure
    if "CIFTI_STRUCTURE_CORTEX_LEFT" in axis_bm.name:
        data["SurfaceL"] = get_surf_data_from_cifti(
            data_cifti, axis_bm, "CIFTI_STRUCTURE_CORTEX_LEFT"
        )
    if "CIFTI_STRUCTURE_CORTEX_RIGHT" in axis_bm.name:
        data["SurfaceR"] = get_surf_data_from_cifti(
            data_cifti, axis_bm, "CIFTI_STRUCTURE_CORTEX_RIGHT"
        )
    if axis_bm.volume_shape is not None:
        data["Volume"] = get_vol_img_from_cifti(data_cifti, axis_bm)
    return data


def decompose_dtseries(
    img: nib.cifti2.Cifti2Image, dtype: Any = np.float32
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """
    Splits a CIFTI dtseries image into surface and volume parts.

    Note: By default, CIFTI file represents samples in columns.

    Args:
        img: A nib.cifti2.Cifti2Image object.
        dtype: Data type of the splitted data.

    Returns:
        A dict contains splitted CIFTI data. The keys are SurfaceL,
        SurfaceR, and Volume.
    """
    return decompose_dscalar(img, dtype=dtype)


def decompose_dlabel(
    img: nib.cifti2.Cifti2Image, dtype: Any = np.int16
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """
    Splits a CIFTI dlabel image into surface and volume parts.

    Args:
        img: A nib.cifti2.Cifti2Image object.
        dtype: Data type of the splitted data.

    Returns:
        A dict contains splitted CIFTI data. The keys are SurfaceL,
        SurfaceR, and Volume.
    """
    return decompose_dscalar(img, dtype=dtype)


def read_dscalar(
    in_file: Union[PathLike, list[PathLike]],
    volume_as_img: bool = False,
    standardize: Optional[Literal["zscore"]] = None,
    dtype: Any = np.float32,
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """Reads CIFTI dscalar files into surface and volume data array.

    Multiple input files will be concatenated along the 1st dim (row).

    Args:
        in_file: A single or a list of CIFTI dscalar files.
        volume_as_img: If true, the volume part in the CIFTI image is
            extracted as a nib.nifti1.Nifti1Image object. If false, it's
            extracted as a numpy array.
        standardize: Standardize each vertex/voxel along the time
            dimension. Valid: zscore.
        dtype: Data type of the returned data.

    Returns:
        A dict contains splitted CIFTI data. The keys are SurfaceL,
        SurfaceR, and Volume. If in_file is a list of filenames, the
        output is a list of dicts.

    Raises:
        TypeError: Input file is not a CIFTI file.
    """

    file_list = [in_file] if not isinstance(in_file, list) else in_file
    # Loop for each in_file
    data = {"SurfaceL": [], "SurfaceR": [], "Volume": []}
    for f in file_list:
        # Read CIFTI file
        img = nib.load(f)
        if not isinstance(img, nib.cifti2.Cifti2Image):
            raise TypeError(f"File {f} is not a CIFTI file.")
        # Decompose CIFTI file
        data_cifti = decompose_dscalar(img, dtype=dtype)
        # Transpose surface data to make row represents samples and col represents features
        # by default, samples are in columns
        if data_cifti["SurfaceL"] is not None:
            data_cifti["SurfaceL"] = data_cifti["SurfaceL"].T
        if data_cifti["SurfaceR"] is not None:
            data_cifti["SurfaceR"] = data_cifti["SurfaceR"].T
        # Standardize if requested
        if standardize == "zscore":
            if data_cifti["SurfaceL"] is not None:
                data_cifti["SurfaceL"] = zscore(data_cifti["SurfaceL"], axis=0)
            if data_cifti["SurfaceR"] is not None:
                data_cifti["SurfaceR"] = zscore(data_cifti["SurfaceR"], axis=0)
            if data_cifti["Volume"] is not None:
                data_cifti["Volume"] = nib.Nifti1Image(
                    zscore(data_cifti["Volume"].get_fdata(dtype=dtype), axis=3),
                    data_cifti["Volume"].affine,
                )
        data["SurfaceL"].append(data_cifti["SurfaceL"])
        data["SurfaceR"].append(data_cifti["SurfaceR"])
        data["Volume"].append(data_cifti["Volume"])
    # Concatenate multiple input data
    if all(i is not None for i in data["SurfaceL"]):
        data["SurfaceL"] = np.vstack(data["SurfaceL"])
    else:
        data["SurfaceL"] = None
    if all(i is not None for i in data["SurfaceR"]):
        data["SurfaceR"] = np.vstack(data["SurfaceR"])
    else:
        data["SurfaceR"] = None
    if all(i is not None for i in data["Volume"]):
        data["Volume"] = nli.concat_imgs(data["Volume"], dtype=dtype)
    else:
        data["Volume"] = None
    # Extract volume data if requested
    if (not volume_as_img) and (data["Volume"] is not None):
        # Make a mask volume image contains voxels defined in CIFTI file
        # (see func: get_vol_img_from_cifti)
        axis_bm = img.header.get_axis(1)
        vox_indices = tuple(axis_bm.voxel[axis_bm.volume_mask].T)
        mask_vol = np.zeros(axis_bm.volume_shape, dtype=np.int16)
        mask_vol[vox_indices] = 1
        mask_img = nib.Nifti1Image(mask_vol, axis_bm.affine)
        # Extract volume data into a 2d array using nilearn's apply_mask function
        if mask_vol.sum() > 0:
            data["Volume"] = nlm.apply_mask(data["Volume"], mask_img)
    return data


def read_dtseries(
    in_file: Union[PathLike, list[PathLike]],
    volume_as_img: bool = False,
    standardize: Optional[Literal["zscore"]] = None,
    dtype: Any = np.float32,
) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
    """Reads CIFTI dtseries files into surface and volume data array.

    Multiple input files will be concatenated along the 1st dim (row).

    Args:
        in_file: A single or a list of CIFTI dtseries files.
        volume_as_img: If true, the volume part in the CIFTI image is
            extracted as a nib.nifti1.Nifti1Image object. If false, it's
            extracted as a numpy array.
        standardize: Standardize each vertex/voxel along the time
            dimension. Valid: zscore.
        dtype: Data type of the returned data.

    Returns:
        A dict contains splitted CIFTI data. The keys are SurfaceL,
        SurfaceR, and Volume. If in_file is a list of filenames, the
        output is a list of dicts.
    """
    return read_dscalar(in_file, volume_as_img=volume_as_img, standardize=standardize, dtype=dtype)


def read_dscalar_roi(
    in_file: Union[PathLike, list[PathLike]],
    roi_mask: Union[
        dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
        list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
    ],
    standardize: Optional[Literal["zscore"]] = None,
    single_data_array: bool = True,
    dtype: Any = np.float32,
) -> Union[
    np.ndarray,
    dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
    list[np.ndarray],
    list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
]:
    """Reads CIFTI dscalar data within ROI.

    This function could read multiple ROI data at once. It's faster than
    a explicit for loop, since this method only reads the whole data
    once. In that case, the ROI data will be in a list instead of a
    single numpy array of dict.

    Args:
        in_file:  A single or a list of CIFTI dscalar files.
        roi_mask: A (list of) ROI mask dict. It is usually generated by
            the 'make_roi_from_spec' function.
        standardize: Standardize each vertex/voxel along the time
            dimension. Valid: zscore.
        single_data_array: If true, concatenating all parts into a
            single numpy array along columns. Order: SurfaceL, SurfaceR,
            Volume.
        dtype: Data type of the returned data.

    Returns:
        Depending on the inputs, the returned ROI data could be in
        several format.
        If the 'single_data_array' option is True (default), the ROI
        data will be contained in a numpy array. If it's False, the ROI
        data will be in a dict like the roi_mask.
        If the 'roi_mask' is a list of ROI mask dict, the data of each
        ROI will be in a list, and the order is the same as the
        'roi_mask'.
        Multiple input file will always be concatenated along the first
        (row) dimension.
    """

    # Read data (a dict with concatenated data)
    ds = read_dscalar(in_file, volume_as_img=True, standardize=standardize, dtype=dtype)
    # Apply ROI masking
    roi_mask = [roi_mask] if not isinstance(roi_mask, list) else roi_mask
    data = []
    for mask in roi_mask:
        data_roi = {"SurfaceL": None, "SurfaceR": None, "Volume": None}
        if mask["SurfaceL"] is not None:
            data_roi["SurfaceL"] = ds["SurfaceL"][:, mask["SurfaceL"].astype(np.bool)].copy()
        if mask["SurfaceR"] is not None:
            data_roi["SurfaceR"] = ds["SurfaceR"][:, mask["SurfaceR"].astype(np.bool)].copy()
        if mask["Volume"] is not None:
            data_roi["Volume"] = nlm.apply_mask(ds["Volume"], mask["Volume"])
        # Combine SurfaceL, SurfaceR and Volume if requested
        if single_data_array:
            data_roi_single = []
            for part in ["SurfaceL", "SurfaceR", "Volume"]:
                if data_roi[part] is not None:
                    data_roi_single.append(data_roi[part])
            data_roi = np.hstack(data_roi_single)
        data.append(data_roi)
    # Extract data from list if there's only one ROI
    if len(data) == 1:
        data = data[0]
    return data


def read_dtseries_roi(
    in_file: Union[PathLike, list[PathLike]],
    roi_mask: dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
    standardize: Optional[Literal["zscore"]] = None,
    single_data_array: bool = True,
    dtype: Any = np.float32,
) -> Union[
    np.ndarray,
    dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
    list[np.ndarray],
    list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
]:
    """Reads CIFTI dtseries data within ROI.

    This function could read multiple ROI data at once. It's faster than
    a explicit for loop, since this method only reads the whole data
    once. In that case, the ROI data will be in a list instead of a
    single numpy array of dict.

    Args:
        in_file:  A single or a list of CIFTI dtseries files.
        roi_mask: A (list of) ROI mask dict. It is usually generated by
            the 'make_roi_from_spec' function.
        standardize: Standardize each vertex/voxel along the time
            dimension. Valid: zscore.
        single_data_array: If true, concatenating all parts into a
            single numpy array along columns. Order: SurfaceL, SurfaceR,
            Volume.
        dtype: Data type of the returned data.

    Returns:
        Depending on the inputs, the returned ROI data could be in
        several format.
        If the 'single_data_array' option is True (default), the ROI
        data will be contained in a numpy array. If it's False, the ROI
        data will be in a dict like the roi_mask.
        If the 'roi_mask' is a list of ROI mask dict, the data of each
        ROI will be in a list, and the order is the same as the
        'roi_mask'.
        Multiple input file will always be concatenated along the first
        (row) dimension.
    """

    return read_dscalar_roi(
        in_file,
        roi_mask,
        standardize=standardize,
        single_data_array=single_data_array,
        dtype=dtype,
    )


def get_vol_img_from_cifti(
    cifti_data: np.ndarray, axis: nib.cifti2.BrainModelAxis
) -> nib.nifti1.Nifti1Image:
    """
    Extracts volume data as a volume image from CIFTI BrainModel.

    Args:
        cifti_data: A 2d numpy array represents brain data in a CIFTI
            file. The BrainModel(vertice/voxel) is in the 2nd dim
            (dim=1).
        axis: A BrainModelAxis object from a CIFTI file.

    Returns:
        A nib.nifti1.Nifti1Image object.

    Raises:
        TypeError: Argument axis is not a nib.cifti2.BrainModelAxis
            object.
    """

    if not isinstance(axis, nib.cifti2.BrainModelAxis):
        raise TypeError("Argument axis is not a nib.cifti2.BrainModelAxis")
    # Find volume voxels
    vol_mask = axis.volume_mask
    vox_indices = tuple(axis.voxel[vol_mask].T)  # ([x0, x1, ...], [y0, ...], [z0, ...])
    # Extract data from volume voxels and make a 4d/3d array
    cifti_data = cifti_data.T[vol_mask]  # Assume brainmodels axis is last, move it to front
    vol_data = np.zeros(axis.volume_shape + cifti_data.shape[1:], dtype=cifti_data.dtype)
    vol_data[vox_indices] = cifti_data  # "Fancy indexing"
    return nib.Nifti1Image(vol_data, axis.affine)


def get_surf_data_from_cifti(
    cifti_data: np.ndarray,
    axis: nib.cifti2.BrainModelAxis,
    surf_name: Literal["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"],
) -> np.ndarray:
    """
    Extracts surface data as a numpy array from CIFTI BrainModel.

    Args:
        cifti_data: A 2d numpy array represents brain data in a CIFTI
            file. The BrainModel(vertice/voxel) is in the 2nd dim
            (dim=1).
        axis: A BrainModelAxis object from a CIFTI file.
        surf_name: The surface name.
            Valid: "CIFTI_STRUCTURE_CORTEX_LEFT",
            "CIFTI_STRUCTURE_CORTEX_RIGHT".

    Returns:
        A numpy array contains surface data.

    Raises:
        TypeError: Argument axis is not a  nib.cifti2.BrainModelAxis
            object.
        ValueError: The surface name is not valid.
    """

    if not isinstance(axis, nib.cifti2.BrainModelAxis):
        raise TypeError("Argument axis is not a nib.cifti2.BrainModelAxis")
    if surf_name not in ["CIFTI_STRUCTURE_CORTEX_LEFT", "CIFTI_STRUCTURE_CORTEX_RIGHT"]:
        raise ValueError(
            f"Surface name {surf_name} is not supported.\n"
            "Valid values: CIFTI_STRUCTURE_CORTEX_LEFT, CIFTI_STRUCTURE_CORTEX_RIGHT"
        )
    # Loop through brain structures
    # Surface name should be CIFTI_STRUCTURE_CORTEX_LEFT or CIFTI_STRUCTURE_CORTEX_RIGHT
    for name, data_indices, model in axis.iter_structures():
        if name == surf_name:
            # Find vertex
            vtx_indices = model.vertex
            # Extract data from surface vertices and make a 2d/1d array
            # assume brainmodels axis is last, move it to front
            cifti_data = cifti_data.T[data_indices]
            surf_data = np.zeros(
                (vtx_indices.max() + 1,) + cifti_data.shape[1:], dtype=cifti_data.dtype
            )
            surf_data[vtx_indices] = cifti_data

            # Returned data has vertices index in the 1st dim (dim=0)
            return surf_data


###################
# Create CIFTI file
###################


def make_dense_scalar_file(
    out_file: PathLike,
    left_surf_data: Optional[np.ndarray] = None,
    right_surf_data: Optional[np.ndarray] = None,
    volume_img: Optional[nib.nifti1.Nifti1Image] = None,
    volume_label_file: Optional[Union[PathLike, nib.nifti1.Nifti1Image]] = None,
    cifti_map_name: Optional[Union[str, list[str]]] = "",
    **kwargs,
) -> Path:
    """Combines surface and volume data to make a CIFTI dscalar file.

    Args:
        out_file: Output CIFTI dscalar file.
        left_surf_data: Left surface data.
        right_surf_file: Right surface data.
        volume_img: Volume NIFTI image.
        volume_label_file: Volume structure label file. It could also be
            a nib.nifti1.Nifti1Image object.
        cifti_map_name: CIFTI image map name. It could be a list of
            string corresponds to each map in the CIFTI file.
        **kwargs: Keyword arguments pass to function
            'assemble_dense_scalar_file'.

    Returns:
        A CIFTI dscalar file.

    Raises:
        ValueError: Left or right surface has incorrect dimensions.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        # write surface/volume data to temporary file
        lh_file, rh_file, vol_file = None, None, None
        if not left_surf_data is None:
            if left_surf_data.ndim == 1:
                lh_img = make_gifti_image(left_surf_data, "CortexLeft")
            elif left_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                lh_img = make_gifti_image(
                    [left_surf_data[i, :] for i in range(left_surf_data.shape[0])], "CortexLeft"
                )
            else:
                ValueError("Argument 'left_surf_data should be a 1d or 2d numpy array.'")
            lh_file = Path(tmp_dir).joinpath("hemi-L.shape.gii")
            lh_img.to_filename(lh_file)
        if not right_surf_data is None:
            if right_surf_data.ndim == 1:
                rh_img = make_gifti_image(right_surf_data, "CortexRight")
            elif right_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                rh_img = make_gifti_image(
                    [right_surf_data[i, :] for i in range(right_surf_data.shape[0])], "CortexRight"
                )
            else:
                ValueError("Argument 'right_surf_data should be a 1d or 2d numpy array.'")
            rh_file = Path(tmp_dir).joinpath("hemi-R.shape.gii")
            rh_img.to_filename(rh_file)
        if not volume_img is None:
            vol_file = Path(tmp_dir).joinpath("volume.nii.gz")
            volume_img.to_filename(vol_file)
        if isinstance(volume_label_file, nib.nifti1.Nifti1Image):
            volume_label_file.to_filename(Path(tmp_dir).joinpath("volume_label.nii.gz"))
            volume_label_file = Path(tmp_dir).joinpath("volume_label.nii.gz")
        # assemble CIFTI file
        out_file = assemble_dense_scalar_file(
            out_file,
            left_surf_file=lh_file,
            right_surf_file=rh_file,
            volume_file=vol_file,
            volume_label_file=volume_label_file,
            cifti_map_name=cifti_map_name,
            **kwargs,
        )
    return out_file


def make_dense_label_file(
    out_file: PathLike,
    label: dict[str, dict[str, Union[str, int, float]]],
    left_surf_data: Optional[np.ndarray] = None,
    right_surf_data: Optional[np.ndarray] = None,
    cifti_map_name: Optional[Union[str, list[str]]] = "",
    **kwargs,
) -> Path:
    """Combines L and R surface data to make a CIFTI dlabel file.

    Args:
        out_file: Output CIFTI dlabel file.
        label: Lookup table of the labels. It should be a dict in
            the format of {label_name: {key:key, red:value, green:value,
            blue:value, alpha:value}}.
        left_surf_data: Left surface data.
        right_surf_file: Right surface data.
        cifti_map_name: CIFTI image map name. It could be a list of
            string corresponds to each map in the CIFTI file.
        **kwargs: Keyword arguments pass to function
            'assemble_dense_scalar_file'.

    Returns:
        A CIFTI dlabel file.

    Raises:
        ValueError: Left or right surface has incorrect dimensions.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        # write surface/volume data to temporary file
        lh_file, rh_file = None, None
        if not left_surf_data is None:
            if left_surf_data.ndim == 1:
                lh_img = make_gifti_label_image(left_surf_data, "CortexLeft", label)
            elif left_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                lh_img = make_gifti_label_image(
                    [left_surf_data[i, :] for i in range(left_surf_data.shape[0])],
                    "CortexLeft",
                    label,
                )
            else:
                ValueError("Argument 'left_surf_data should be a 1d or 2d numpy array.'")
            lh_file = Path(tmp_dir).joinpath("hemi-L.shape.gii")
            lh_img.to_filename(lh_file)
        if not right_surf_data is None:
            if right_surf_data.ndim == 1:
                rh_img = make_gifti_label_image(right_surf_data, "CortexRight", label)
            elif right_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                rh_img = make_gifti_label_image(
                    [right_surf_data[i, :] for i in range(right_surf_data.shape[0])],
                    "CortexRight",
                    label,
                )
            else:
                ValueError("Argument 'right_surf_data should be a 1d or 2d numpy array.'")
            rh_file = Path(tmp_dir).joinpath("hemi-R.shape.gii")
            rh_img.to_filename(rh_file)
        # assemble CIFTI file
        out_file = assemble_dense_label_file(
            out_file,
            left_surf_file=lh_file,
            right_surf_file=rh_file,
            cifti_map_name=cifti_map_name,
            **kwargs,
        )
    return out_file


def make_dense_timeseries_file(
    out_file: PathLike,
    timestep: float,
    left_surf_data: Optional[np.ndarray] = None,
    right_surf_data: Optional[np.ndarray] = None,
    volume_img: Optional[nib.nifti1.Nifti1Image] = None,
    volume_label_file: Optional[Union[PathLike, nib.nifti1.Nifti1Image]] = None,
    **kwargs,
) -> Path:
    """Combines surface and volume data to make a CIFTI dtseries file.

    Args:
        out_file: Output CIFTI dtseries file.
        timestep: Repetition time (TR).
        left_surf_data: Left surface data.
        right_surf_file: Right surface data.
        volume_img: Volume NIFTI image.
        volume_label_file: Volume structure label file. It could also be
            a nib.nifti1.Nifti1Image object.
        **kwargs: Keyword arguments pass to function
            'assemble_dense_dtseries_file'.

    Returns:
        A CIFTI dtseries file.

    Raises:
        ValueError: Left or right surface has incorrect dimensions.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:
        # write surface/volume data to temporary file
        lh_file, rh_file, vol_file = None, None, None
        if not left_surf_data is None:
            if left_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                lh_img = make_gifti_image(
                    [left_surf_data[i, :] for i in range(left_surf_data.shape[0])], "CortexLeft"
                )
            else:
                ValueError("Argument 'left_surf_data should be a 2d numpy array.'")
            lh_file = Path(tmp_dir).joinpath("hemi-L.func.gii")
            lh_img.to_filename(lh_file)
        if not right_surf_data is None:
            if right_surf_data.ndim == 2:
                # assume maps in the 1st dimension
                rh_img = make_gifti_image(
                    [right_surf_data[i, :] for i in range(right_surf_data.shape[0])], "CortexRight"
                )
            else:
                ValueError("Argument 'right_surf_data should be a 2d numpy array.'")
            rh_file = Path(tmp_dir).joinpath("hemi-R.func.gii")
            rh_img.to_filename(rh_file)
        if not volume_img is None:
            vol_file = Path(tmp_dir).joinpath("volume.nii.gz")
            volume_img.to_filename(vol_file)
        if isinstance(volume_label_file, nib.nifti1.Nifti1Image):
            volume_label_file.to_filename(Path(tmp_dir).joinpath("volume_label.nii.gz"))
            volume_label_file = Path(tmp_dir).joinpath("volume_label.nii.gz")
        # assemble CIFTI file
        out_file = assemble_dense_timeseries_file(
            out_file,
            timestep,
            left_surf_file=lh_file,
            right_surf_file=rh_file,
            volume_file=vol_file,
            volume_label_file=volume_label_file,
            **kwargs,
        )
    return out_file


def assemble_dense_scalar_file(
    out_file: PathLike,
    left_surf_file: Optional[PathLike] = None,
    right_surf_file: Optional[PathLike] = None,
    left_roi_file: Optional[PathLike] = None,
    right_roi_file: Optional[PathLike] = None,
    volume_file: Optional[PathLike] = None,
    volume_label_file: Optional[PathLike] = None,
    cifti_map_name: Optional[Union[str, list[str]]] = "",
) -> Path:
    """Combines surface and volume files to make a CIFTI dscalar file.

    Args:
        out_file: Output CIFTI dscalar file.
        left_surf_file: Left surface GIFTI file.
        right_surf_file: Right surface GIFTI file.
        left_roi_file: Left surface mask file.
        right_roi_file: Right surfce mask file.
        volume_file: Volume NIFTI file.
        volume_label_file: Volume structure label file.
        cifti_map_name: CIFTI image map name. It could be a list of
            string corresponds to each map in the CIFTI file.

    Returns:
        A CIFTI dscalar file.

    Raises:
        ValueError: The volume_label_file is None when volume_file is
            specified.
    """

    cmd = f"wb_command -disable-provenance -cifti-create-dense-scalar {out_file} "
    if left_surf_file is not None:
        cmd += f"-left-metric {left_surf_file} "
    if left_roi_file is not None:
        cmd += f"-roi-left {left_roi_file} "
    if right_surf_file is not None:
        cmd += f"-right-metric {right_surf_file} "
    if right_roi_file is not None:
        cmd += f"-roi-right {right_roi_file} "
    if volume_file is not None:
        if volume_label_file is not None:
            cmd += f"-volume {volume_file} {volume_label_file}"
        else:
            raise ValueError("If volume_file is provided, volume_label_file is also required.")
    print(f"Creating dense scalar file: {out_file} ...", flush=True)
    run_cmd(cmd)
    # Set metadata
    cmd = f"wb_command -disable-provenance -set-map-names {out_file} "
    if isinstance(cifti_map_name, str):
        cifti_map_name = [cifti_map_name]
    for i in range(len(cifti_map_name)):
        cmd += f"-map {i+1} {cifti_map_name[i]} "
    run_cmd(cmd)
    return Path(out_file)


def assemble_dense_label_file(
    out_file: PathLike,
    left_surf_file: Optional[PathLike] = None,
    right_surf_file: Optional[PathLike] = None,
    left_roi_file: Optional[PathLike] = None,
    right_roi_file: Optional[PathLike] = None,
    cifti_map_name: Optional[Union[str, list[str]]] = "",
) -> Path:
    """Combines L and R surface files to make a CIFTI dlabel file.

    Args:
        out_file: Output CIFTI dlabel file.
        left_surf_file: Left surface GIFTI file.
        right_surf_file: Right surface GIFTI file.
        left_roi_file: Left surface mask file.
        right_roi_file: Right surfce mask file.
        cifti_map_name: CIFTI image map name. It could be a list of
            string corresponds to each map in the CIFTI file.

    Returns:
        A CIFTI dlabel file.

    Raises:
        ValueError: Left or right ROI file is not found.
    """

    if (left_surf_file is not None) and (left_roi_file is None):
        raise ValueError("If left_surf_file is provided, left_roi_file is also required.")
    if (right_surf_file is not None) and (right_roi_file is None):
        raise ValueError("If right_surf_file is provided, right_roi_file is also required.")

    cmd = f"wb_command -logging SEVERE -disable-provenance -cifti-create-label {out_file} "
    if left_surf_file is not None:
        cmd += f"-left-label {left_surf_file} -roi-left {left_roi_file} "
    if right_surf_file is not None:
        cmd += f"-right-label {right_surf_file} -roi-right {right_roi_file}"
    print(f"Creating dense label file: {out_file} ...", flush=True)
    run_cmd(cmd)
    # Set metadata
    cmd = f"wb_command -disable-provenance -set-map-names {out_file} "
    if isinstance(cifti_map_name, str):
        cifti_map_name = [cifti_map_name]
    for i in range(len(cifti_map_name)):
        cmd += f"-map {i+1} {cifti_map_name[i]} "
    run_cmd(cmd)
    return Path(out_file)


def assemble_dense_timeseries_file(
    out_file: PathLike,
    timestep: float,
    left_surf_file: Optional[PathLike] = None,
    right_surf_file: Optional[PathLike] = None,
    left_roi_file: Optional[PathLike] = None,
    right_roi_file: Optional[PathLike] = None,
    volume_file: Optional[PathLike] = None,
    volume_label_file: Optional[PathLike] = None,
) -> Path:
    """Combines surface and volume files to make a CIFTI dtseries file.

    Args:
        out_file: Output CIFTI dtseries file.
        timestep: Repetition time (TR).
        left_surf_file: Left surface GIFTI file.
        right_surf_file: Right surface GIFTI file.
        left_roi_file: Left surface mask file.
        right_roi_file: Right surfce mask file.
        volume_file: Volume NIFTI file.
        volume_label_file: Volume structure label file.

    Returns:
        A CIFTI dtseries file.

    Raises:
        ValueError: The volume_label_file is None when volume_file is
            specified.
    """

    cmd = (
        "wb_command -disable-provenance -cifti-create-dense-timeseries "
        f"{out_file} -timestep {timestep} "
    )
    if left_surf_file is not None:
        cmd += f"-left-metric {left_surf_file} "
    if left_roi_file is not None:
        cmd += f"-roi-left {left_roi_file} "
    if right_surf_file is not None:
        cmd += f"-right-metric {right_surf_file} "
    if right_roi_file is not None:
        cmd += f"-roi-right {right_roi_file} "
    if volume_file is not None:
        if volume_label_file is not None:
            cmd += f"-volume {volume_file} {volume_label_file}"
        else:
            raise ValueError("If volume_file is provided, volume_label_file is also required.")
    print(f"Creating dense timeseries file: {out_file} ...", flush=True)
    run_cmd(cmd)
    return Path(out_file)


#######################
# Manipulate CIFTI file
#######################


def concat_dtseries(file_list: list[PathLike], out_file: PathLike) -> Path:
    """Concatenates a list of CIFTI dtseries files to a single file.

    Args:
        file_list: A list of CIFTI dtseries files.
        out_file: Output CIFTI dtseries file.

    Returns:
        A CIFTI dtseries file.
    """

    cmd = f"wb_command -disable-provenance -cifti-merge {out_file} "
    for i in file_list:
        cmd += f"-cifti {i} "
    run_cmd(cmd)
    return Path(out_file)


def concat_dscalar(file_list: list[PathLike], out_file: PathLike) -> Path:
    """Concatenates a list of CIFTI dscalar files to a single file.

    Args:
        file_list: A list of CIFTI dscalar files.
        out_file: Output CIFTI dscalar file.

    Returns:
        A CIFTI dscalar file.
    """
    return concat_dtseries(file_list, out_file)


def split_dtseries(
    in_file: PathLike,
    left_surf_out_file: Optional[PathLike] = None,
    right_surf_out_file: Optional[PathLike] = None,
    volume_out_file: Optional[PathLike] = None,
    volume_mask_out_file: Optional[PathLike] = None,
) -> list[Optional[Path]]:
    """
    Splits a CIFTI dtseries file to GIFTI and NIFTI files.

    Args:
        left_surf_out_file: Left surface output GIFTI file.
        right_surf_out_file: Right surface output GIFTI file.
        volume_out_file: Volume output file.
        volume_mask_out_file: Volume structure label output file.

    Returns:
        A list of files. The files are in the following order: left
        surface GIFTI, right surface GIFTI, and subcortical NIFTI
        (Optional: subcortrical mask NIFTI). Nonexistent part is None.

    Raises:
        ValueError: None of the output file is specified.
    """

    if (
        (left_surf_out_file is None)
        and (right_surf_out_file is None)
        and (volume_out_file is None)
    ):
        raise ValueError("At least one output file is required.")
    out_file = [None, None, None, None]
    lh_cmd, rh_cmd, volume_cmd, volume_mask_cmd = "", "", "", ""
    if left_surf_out_file is not None:
        lh_cmd = f"-metric CORTEX_LEFT {left_surf_out_file}"
        out_file[0] = Path(left_surf_out_file)
    if right_surf_out_file is not None:
        rh_cmd = f"-metric CORTEX_RIGHT {right_surf_out_file}"
        out_file[1] = Path(right_surf_out_file)
    if volume_out_file is not None:
        volume_cmd = f"-volume-all {volume_out_file}"
        out_file[2] = Path(volume_out_file)
    if volume_mask_out_file is not None:
        volume_mask_cmd = f"-roi {volume_mask_out_file}"
        out_file[3] = Path(volume_mask_out_file)
    cmd = (
        f"wb_command -disable-provenance -cifti-separate {in_file} COLUMN "
        f"{lh_cmd} {rh_cmd} {volume_cmd} {volume_mask_cmd}"
    )
    run_cmd(cmd)
    return out_file


def split_dscalar(
    in_file: PathLike,
    left_surf_out_file: Optional[PathLike] = None,
    right_surf_out_file: Optional[PathLike] = None,
    volume_out_file: Optional[PathLike] = None,
    volume_mask_out_file: Optional[PathLike] = None,
) -> list[Optional[Path]]:
    """
    Splits a CIFTI dscalar file to GIFTI and NIFTI files.

    Args:
        left_surf_out_file: Left surface output GIFTI file.
        right_surf_out_file: Right surface output GIFTI file.
        volume_out_file: Volume output file.
        volume_mask_out_file: Volume structure label output file.

    Returns:
        A list of files. The files are in the following order: left
        surface GIFTI, right surface GIFTI, and subcortical NIFTI
        (Optional: subcortrical mask NIFTI). Nonexistent part is None.

    Raises:
        ValueError: None of the output file is specified.
    """

    out_file = split_dtseries(
        in_file,
        left_surf_out_file=left_surf_out_file,
        right_surf_out_file=right_surf_out_file,
        volume_out_file=volume_out_file,
        volume_mask_out_file=volume_mask_out_file,
    )
    return out_file


def extract_dscalar_map(in_file: PathLike, out_file: PathLike, mapname: str) -> Path:
    """
    Extracts a map from a CIFTI dscalar file based on map name.

    Args:
        in_file: CIFTI dscalar file.
        out_file: Output CIFTI dscalar file.
        mapname: Name of the to be extracted map.

    Returns:
        A CIFTI dscalar file contains required map.

    Raises:
        ValueError: Map name is not found in the input CIFTI file.
    """

    # Get map names from input file
    name_list = get_dscalar_map_name(in_file)
    if mapname not in name_list:
        raise ValueError(f"Map '{mapname}' is not found in {in_file}.")
    # Extract selected map
    idx = name_list.index(mapname) + 1
    cmd = (
        f"wb_command -disable-provenance -cifti-math 'a' {out_file} "
        f"-var 'a' {in_file} -select 1 {idx}"
    )
    run_cmd(cmd, print_output=False)
    # Set proper map name
    set_dscalar_map_name(out_file, mapname, 1)
    return Path(out_file)


def get_dscalar_map_name(in_file: PathLike) -> list[str]:
    """
    Gets map name from a CIFTI dscalar file.

    Args:
        in_file: CIFTI dscalar file.

    Returns:
        A list of map names of the input dscalar file.
    """

    cmd = ["wb_command", "-file-information", in_file, "-only-map-names"]
    map_name = run_cmd(cmd, print_output=False).stdout.split()
    return map_name


def set_dscalar_map_name(in_file: PathLike, map_name: str, map_index: int) -> Path:
    """
    Sets CIFTI dscalar map name.

    Args:
        in_file: CIFTI dscalar file.
        map_name: Map name to be set.
        map_index: Index of the dscalar map.

    Returns:
        A CIFTI dscalar file. This is an inplace operation and the
        output filename is the same as the input.
    """

    cmd = (
        f"wb_command -disable-provenance -set-map-names {in_file} "
        f"-map {str(map_index)} {map_name}"
    )
    run_cmd(cmd)
    return Path(in_file)
