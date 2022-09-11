#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to GIFTI file manipulation."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from typing import Optional, Union, Literal, Any
from pathlib import Path
import numpy as np
import nibabel as nib

from ..utils.shell import run_cmd
from ..utils.typing import PathLike


####################
# Create GIFTI image
####################


def make_gifti_image(
    data: Union[np.ndarray, list[np.ndarray]],
    structure: str,
    intent: str = "NIFTI_INTENT_NONE",
    datatype: str = "NIFTI_TYPE_FLOAT32",
    kw_args_image: dict[str, Any] = {},
    kw_args_darray: dict[str, Any] = {},
) -> nib.gifti.gifti.GiftiImage:
    """
    Makes GIFTI image.

    Args:
        data: Data array in GIFTI file. It could be a numpy array or a
            list of numpy array.
        structure: Primary structure metadata of GIFTI image. Usually,
            it could be Cortex_Left or Cortex_Right.
        intent: GIFTI file intent type.
        datatype: GIFTI file data type.
        kw_args_image: Additional keyword arguments pass to nibabel
            GiftiImage class. Image metadata should be stored in an item
            which key is 'meta'.
        kw_args_darray: Additional keyword arguments pass to nibabel
            GiftiDataArray class. Data array metadata should be stored
            in an item which key is 'meta'.

    Returns:
        A nib.gifti.gifti.GiftiImage object.
    """

    # Conform input
    data = [data] if not isinstance(data, list) else data
    # Set image metadata
    if "meta" in kw_args_image.keys():
        meta_dict = kw_args_image["meta"].metadata
        meta_dict.update({"AnatomicalStructurePrimary": structure})
    else:
        meta_dict = {"AnatomicalStructurePrimary": structure}
    kw_args_image["meta"] = nib.gifti.gifti.GiftiMetaData().from_dict(meta_dict)
    # Initialize GIFTI image object
    img = nib.gifti.gifti.GiftiImage(**kw_args_image)
    # Add data array
    for da in data:
        da = nib.gifti.gifti.GiftiDataArray(
            data=da, intent=intent, datatype=datatype, **kw_args_darray
        )
        img.add_gifti_data_array(da)
    return img


def make_gifti_label_image(
    data: Union[np.ndarray, list[np.ndarray]],
    structure: str,
    label: dict[str, dict[str, Union[str, int, float]]],
    kw_args_image: dict[str, Any] = {},
    kw_args_darray: dict[str, Any] = {},
) -> nib.gifti.gifti.GiftiImage:
    """
    Makes GIFTI label image.

    Args:
        data: Data array in GIFTI file. It could be a numpy array or a
            list of numpy array.
        structure: Primary structure metadata of GIFTI image. Usually,
            it could be CortexLeft or CortexRight.
        label: Lookup table of the labels. It should be a dict in
            the format of {label_name: {key:key, red:value, green:value,
            blue:value, alpha:value}}.
        kw_args_image: Additional keyword arguments pass to nibabel
            GiftiImage class. Image metadata should be stored in an item
            which key is 'meta'.
        kw_args_darray: Additional keyword arguments pass to nibabel
            GiftiDataArray class. Data array metadata should be stored
            in an item which key is 'meta'.

    Returns:
        A nib.gifti.gifti.GiftiImage object.
    """

    # Conform input
    data = [data] if not isinstance(data, list) else data
    # Set image metadata
    if "meta" in kw_args_image.keys():
        meta_dict = kw_args_image["meta"].metadata
        meta_dict.update({"AnatomicalStructurePrimary": structure})
    else:
        meta_dict = {"AnatomicalStructurePrimary": structure}
    kw_args_image["meta"] = nib.gifti.gifti.GiftiMetaData().from_dict(meta_dict)
    # Initialize GIFTI image object
    img = nib.gifti.gifti.GiftiImage(**kw_args_image)
    # Set LabelTable
    label_table = nib.gifti.gifti.GiftiLabelTable()
    for label_name, attr in label.items():
        label = nib.gifti.gifti.GiftiLabel(**attr)
        label.label = label_name
        label_table.labels.append(label)
    img.labeltable = label_table
    # Add data array
    for da in data:
        da = nib.gifti.gifti.GiftiDataArray(
            data=da, intent="NIFTI_INTENT_LABEL", datatype="NIFTI_TYPE_INT32", **kw_args_darray
        )
        img.add_gifti_data_array(da)
    return img


def sanitize_gii_metadata(
    in_file: PathLike,
    out_file: PathLike,
    gim_atr: dict[str, str] = {},
    gim_meta: dict[str, str] = {},
    da_atr: dict[str, str] = {},
    da_meta: dict[str, str] = {},
    clean_provenance: bool = False,
) -> Path:
    """
    Cleanup metadata and validate GIFTI file.

    Args:
        in_file: GIFTI file.
        out_file: Output GIFTI file.
        gim_atr: GIFTI image attribute. It could be used for adding new
            attributes or modifying existed ones.
        gim_meta: GIFTI image metadata. It could be used for adding new
            attributes or modifying existed ones.
        da_atr: GIFTI data array attribute. It could be used for adding
            new attributes or modifying existed ones.
        da_meta: GIFTI data array metadata. It could be used for adding
            new attributes or modifying existed ones.
        clean_provenance: Remove provenance data (usually generated by
            HCP Workbench).

    Returns:
        A GIFTI file. This is an inplace operation and the output
        filename is the same as the input.
    """

    # Fix Gifti image metadata `Version`
    # When gifti file is processed by wb_command, the field `version`
    # in Gifti image metadata will be set to 1. This will cause error
    # when loading data to Freeview. Fix by setting it to 1.0.
    sanitize_cmd = [
        "gifti_tool",
        "-infile",
        in_file,
        "-write_gifti",
        out_file,
        "-mod_gim_atr",
        "Version",
        "1.0",
    ]
    # Replace user specified fields
    if gim_atr:
        for key, value in gim_atr.items():
            sanitize_cmd += ["-mod_gim_atr", key, value]
    if gim_meta:
        for key, value in gim_meta.items():
            sanitize_cmd += ["-mod_gim_meta", key, value]
    if da_atr:
        for key, value in da_atr.items():
            sanitize_cmd += ["-mod_DA_atr", key, value]
    if da_meta:
        for key, value in da_meta.items():
            sanitize_cmd += ["-mod_DA_meta", key, value]
    run_cmd(sanitize_cmd)
    # Cleanup provenance metadata added by wb_command
    if clean_provenance:
        wb_gim_meta = {
            "ProgramProvenance": "",
            "Provenance": "",
            "WorkingDirectory": "",
        }
        for key, value in wb_gim_meta.items():
            sanitize_cmd += ["-mod_gim_meta", key, value]
    # Verify output gifti file
    run_cmd(["gifti_tool", "-infile", out_file, "-gifti_test"])
    return Path(out_file)


#######################
# Manipulate GIFTI file
#######################


def resample_surface(
    surf_file: PathLike,
    current_sphere_file: PathLike,
    target_sphere_file: PathLike,
    out_file: PathLike,
) -> Path:
    """Resamples surface mesh to target space.

    Args:
        surf_file: Surface mesh file. (e.g., native mesh in native
            space)
        current_sphere_file: Sphere surface file with the mesh that the
            surf_file is currently on. (e.g., native mesh in fsLR space)
        target_sphere_file: Sphere surface file that is in register with
            current_sphere_file and has the desired output mesh. (e.g.,
            164k mesh in fsLR space)
        out_file: Output surface mesh file.

    Returns:
        A resampled surface mesh file.
    """

    # Resample
    run_cmd(
        f"wb_command -disable-provenance -surface-resample {surf_file} "
        f"{current_sphere_file} {target_sphere_file} BARYCENTRIC {out_file}"
    )
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)

    return Path(out_file)


def resample_metric(
    metric_file: PathLike,
    current_sphere_file: PathLike,
    target_sphere_file: PathLike,
    out_file: PathLike,
    current_area_surf_file: Optional[PathLike] = None,
    target_area_surf_file: Optional[PathLike] = None,
    roi_file: Optional[PathLike] = None,
    resample_method: Literal["ADAP_BARY_AREA", "BARYCENTRIC"] = "ADAP_BARY_AREA",
) -> Path:
    """Resamples surface metric to target space.

    Args:
        metric_file: Surface metric file. (e.g., native sulc in native
            space)
        current_sphere_file: Sphere surface file with the mesh that the
            surf_file is currently on. (e.g., native mesh in fsLR space)
        target_sphere_file: Sphere surface file that is in register with
            current_sphere_file and has the desired output mesh. (e.g.,
            164k mesh in fsLR space)
        out_file: Output surface metric file.
        current_area_surf_file: Surface used for vertex area correction.
            The mesh of this surface should match current_sphere_file.
            (e.g., midthickness in native space with native mesh)
        target_area_surf_file: Surface used for vertex area correction.
            The mesh of this surface should match target_sphere_file.
            (e.g., midthickness in fsLR space with 164k mesh)
        roi_file: Surface mask file applies to current_sphere_file.
        resample_method: Resample method. ADAP_BARY_AREA or BARYCENTRIC.

    Returns:
        A resampled surface metric file.

    Raises:
        ValueError: Unrecognized resample method.
    """

    # Parse resample method
    if resample_method not in ["ADAP_BARY_AREA", "BARYCENTRIC"]:
        raise ValueError("Unrecognized resample method. Valid: ADAP_BARY_AREA, BARYCENTRIC.")

    # Resample
    cmd = (
        f"wb_command -disable-provenance -metric-resample "
        f"{metric_file}  {current_sphere_file} {target_sphere_file} "
        f"{resample_method} {out_file} "
    )
    if resample_method == "ADAP_BARY_AREA":
        cmd += f"-area-surfs {current_area_surf_file} {target_area_surf_file} "
    if resample_method == "BARYCENTRIC":
        cmd += f"-largest "
    if roi_file:
        cmd += f"-current-roi {roi_file}"
    run_cmd(cmd)
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)

    return Path(out_file)


def resample_label(
    label_file: PathLike,
    current_sphere_file: PathLike,
    target_sphere_file: PathLike,
    out_file: PathLike,
    current_area_surf_file: Optional[PathLike] = None,
    target_area_surf_file: Optional[PathLike] = None,
    resample_method: Literal["ADAP_BARY_AREA", "BARYCENTRIC"] = "ADAP_BARY_AREA",
) -> Path:
    """Resamples surface label to target space.

    Args:
        label_file: Surface label file. (e.g., native aparc in native
            space)
        current_sphere_file: Sphere surface file with the mesh that the
            surf_file is currently on. (e.g., native mesh in fsLR space)
        target_sphere_file: Sphere surface file that is in register with
            current_sphere_file and has the desired output mesh. (e.g.,
            164k mesh in fsLR space)
        out_file: Output surface label file.
        current_area_surf_file: Surface used for vertex area correction.
            The mesh of this surface should match current_sphere_file.
            (e.g., midthickness in native space with native mesh)
        target_area_surf_file: Surface used for vertex area correction.
            The mesh of this surface should match target_sphere_file.
            (e.g., midthickness in fsLR space with 164k mesh)
        resample_method: Resample method. ADAP_BARY_AREA or BARYCENTRIC.

    Returns:
        A resampled surface label file.

    Raises:
        ValueError: Unrecognized resample method.
    """

    # Parse resample method
    if resample_method not in ["ADAP_BARY_AREA", "BARYCENTRIC"]:
        raise ValueError("Unrecognized resample method. Valid: ADAP_BARY_AREA, BARYCENTRIC.")

    # Resample
    cmd = (
        f"wb_command -disable-provenance -label-resample "
        f"{label_file}  {current_sphere_file} {target_sphere_file} "
        f"{resample_method} {out_file} "
    )
    if resample_method == "ADAP_BARY_AREA":
        cmd += f"-area-surfs {current_area_surf_file} {target_area_surf_file}"
    if resample_method == "BARYCENTRIC":
        cmd += f"-largest "
    run_cmd(cmd)
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)

    return Path(out_file)
