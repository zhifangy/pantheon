#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to fit GLM."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Optional
from pathlib import Path
import shutil
import tempfile
import nibabel as nib

from ..image.cifti import split_dtseries
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def fit_3dREMLfit_cifti(
    func_file: PathLike,
    design_matrix_file: PathLike,
    out_dir: PathLike,
    left_roi_file: Optional[PathLike] = None,
    right_roi_file: Optional[PathLike] = None,
    volume_label_file: Optional[PathLike] = None,
    prefix: Optional[str] = None,
    extra_3dremlfit_args: str = "-tout -rout -noFDR -nobout -quiet",
    debug: bool = False,
) -> Path:
    """Fits GLM with AFNI's 3dREMLfit on CIFTI file.

    Args:
        func_file: A CIFTI dtseries/dscalar file.
        design_matrix_file: Design matrix file in AFNI format.
        out_dir: Directory to store output files.
        left_roi_file: Left surface mask file. Optional.
        right_roi_file: Right surfce mask file. Optional.
        volume_label_file: Volume structure label file. This file is
            required if the input CIFTI file has volume part.
        prefix: The output filename prefix (before .dscalar.nii).
            If None, use default names.
        extra_3dremlfit_args: Extra arguments pass to AFNI's 3dREMLFit
            program.
        debug: If true, save intermediate files to fitted_bucket folder
            inside the out_dir.

    Returns:
        A CIFTI file dscalar contains all outputs from 3dREMLFit.

    Raises:
        ValueError: The volume_label_file is None when there's volume
            part in the input CIFTI file.
    """

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:

        # Parse prefix
        func_prefix = "func" if prefix is None else prefix
        model_prefix = "fitted" if prefix is None else prefix

        # Check which part presents in the CIFTI file and create
        # filenames for splitted parts
        left_surf_file, right_surf_file, volume_file, volume_mask_file = None, None, None, None
        axis_bm = nib.load(func_file).header.get_axis(1)
        if "CIFTI_STRUCTURE_CORTEX_LEFT" in axis_bm.name:
            left_surf_file = Path(tmp_dir, f"{func_prefix}_hemi-L.func.gii")
        if "CIFTI_STRUCTURE_CORTEX_RIGHT" in axis_bm.name:
            right_surf_file = Path(tmp_dir, f"{func_prefix}_hemi-R.func.gii")
        if axis_bm.volume_shape is not None:
            if volume_label_file is None:
                raise ValueError(
                    "There is volume part in the input CIFTI file. "
                    "A volume_label_file is required"
                )
            volume_file = Path(tmp_dir, f"{func_prefix}_volume.nii.gz")
            volume_mask_file = Path(tmp_dir, f"{func_prefix}_volume_mask.nii.gz")
        # Split cifti file to left/right surfaces and volume image
        # index 0: left surface; index 1: right surface; index2: volume
        # index 3: volume mask
        _ = split_dtseries(
            func_file,
            left_surf_out_file=left_surf_file,
            right_surf_out_file=right_surf_file,
            volume_out_file=volume_file,
            volume_mask_out_file=volume_mask_file,
        )

        # Fit GLM using AFNI's 3dREMLfit
        out_file = fit_3dREMLfit_cifti_separate(
            design_matrix_file,
            out_dir,
            left_surf_file=left_surf_file,
            right_surf_file=right_surf_file,
            volume_file=volume_file,
            volume_mask_file=volume_mask_file,
            left_roi_file=left_roi_file,
            right_roi_file=right_roi_file,
            volume_label_file=volume_label_file,
            prefix=model_prefix,
            extra_3dremlfit_args=extra_3dremlfit_args,
            debug=debug,
        )
        if debug:
            debug_dir = Path(out_dir).joinpath("debug", model_prefix)
            shutil.copytree(tmp_dir, debug_dir, dirs_exist_ok=True)

    return out_file


def fit_3dREMLfit_cifti_separate(
    design_matrix_file: PathLike,
    out_dir: PathLike,
    left_surf_file: Optional[PathLike] = None,
    right_surf_file: Optional[PathLike] = None,
    volume_file: Optional[PathLike] = None,
    volume_mask_file: Optional[PathLike] = None,
    left_roi_file: Optional[PathLike] = None,
    right_roi_file: Optional[PathLike] = None,
    volume_label_file: Optional[PathLike] = None,
    prefix: Optional[str] = None,
    extra_3dremlfit_args: str = "-tout -rout -noFDR -nobout -quiet",
    debug: bool = False,
) -> Path:
    """Fits GLM with AFNI's 3dREMLfit on CIFTI file.

    This function fits models on each part of the CIFTI file, which are
    specified separately. It is useful when fitting multiple models on
    the same data, for example, singletrial responses estimation.

    Args:
        design_matrix_file: Design matrix file in AFNI format.
        out_dir: Directory to store output files.
        left_surf_file: Left surface GIFTI file. Optional.
        right_surf_file: Right surface GIFTI file. Optional.
        volume_file: Volume NIFTI file. Optional.
        volume_mask_file: Volume mask file of volume_file. Optional.
        left_roi_file: Left surface mask file. Optional.
        right_roi_file: Right surface mask file. Optional.
        volume_label_file: Volume structure label file. This file is
            required if the input CIFTI file has volume part.
        prefix: The output filename prefix (before .dscalar.nii).
            If None, use default names.
        extra_3dremlfit_args: Extra arguments pass to AFNI's 3dREMLFit
            program.
        debug: If true, save intermediate files to fitted_bucket folder
            inside the out_dir.

    Returns:
        A CIFTI file dscalar contains all outputs from 3dREMLFit.

    Raises:
        ValueError: None of the input file is specified.
        ValueError: Input file's format is incorrect.
        ValueError: The volume_label_file is None when volume_file is
            specified.
    """

    if (left_surf_file is None) and (right_surf_file is None) and (volume_file is None):
        raise ValueError("At least one input file is required.")
    if (left_surf_file is not None) and (not Path(left_surf_file).name.endswith(".gii")):
        raise ValueError("Argument left_surf_file should be a GIFTI file.")
    if (right_surf_file is not None) and (not Path(right_surf_file).name.endswith(".gii")):
        raise ValueError("Argument right_surf_file should be a GIFTI file.")
    if (volume_file is not None) and (not Path(volume_file).name.endswith(("nii.gz", ".nii"))):
        raise ValueError("Argument volume_file should be a NIFTI file.")
    if (volume_file is not None) and (volume_label_file is None):
        raise ValueError("When volume_file is specified, the volume_label_file is required.")

    # Parse prefix
    prefix = "fitted" if prefix is None else prefix

    with tempfile.TemporaryDirectory(dir=out_dir) as tmp_dir:

        # Fit GLM using AFNI's 3dREMLfit
        bucket_file = []
        # surface
        for i, j, hemi in zip(
            [left_surf_file, right_surf_file], [left_roi_file, right_roi_file], ["L", "R"]
        ):
            if i is not None:
                mask = f"-mask {j}" if j is not None else ""
                out_file = Path(tmp_dir, f"{prefix}_hemi-{hemi}_bucket.func.gii")
                cmd = (
                    f"3dREMLfit -input {i} {mask} -matrix {design_matrix_file} "
                    f"-Rbuck {out_file} {extra_3dremlfit_args}"
                )
                run_cmd(cmd, cwd=tmp_dir)
                bucket_file.append(out_file)
            else:
                bucket_file.append(None)
        # volume
        if volume_file is not None:
            mask = f"-mask {volume_mask_file}" if volume_mask_file is not None else ""
            out_file = Path(tmp_dir, f"{prefix}_volume_bucket.nii.gz")
            cmd = (
                f"3dREMLfit -input {volume_file} {mask} -matrix {design_matrix_file} "
                f"-Rbuck {out_file} {extra_3dremlfit_args} "
            )
            run_cmd(cmd, cwd=tmp_dir)
            bucket_file.append(out_file)
        else:
            bucket_file.append(None)

        # Merge surface and volume data back to CIFTI
        out_file = Path(out_dir).joinpath(f"{prefix}_bucket.dscalar.nii")
        cmd = f"wb_command -disable-provenance -cifti-create-dense-scalar {out_file} "
        if bucket_file[0] is not None:
            cmd += f"-left-metric {bucket_file[0]} "
            if left_roi_file is not None:
                cmd += f"-roi-left {left_roi_file} "
        if bucket_file[1] is not None:
            cmd += f"-right-metric {bucket_file[1]} "
            if right_roi_file is not None:
                cmd += f"-roi-right {right_roi_file} "
        if bucket_file[2] is not None:
            cmd += f"-volume {bucket_file[2]} {volume_label_file}"
        run_cmd(cmd)
        if debug:
            debug_dir = Path(out_dir).joinpath("debug", prefix)
            shutil.copytree(tmp_dir, debug_dir, dirs_exist_ok=True)

    return out_file
