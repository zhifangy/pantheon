#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:
#

from __future__ import annotations
from typing import Optional, Union
from pathlib import Path
import shutil
import tempfile
import numpy as np

from ..image.gifti import sanitize_gii_metadata
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def find_good_voxel(
    func_file: PathLike,
    ribbon_file: PathLike,
    out_file: PathLike,
    neighborhood_smoothing: Union[float, int] = 5,
    ci_limit: Union[float, int] = 0.5,
    debug: bool = False,
) -> Path:
    """Finds good voxels based on coefficient of variation.

    Args:
        func_file: Functional image file.
        ribbon_file: Cortex ribbon file.
        out_file: Output good voxels mask file.
        neighborhood_smoothing: Spatial smoothing kernal sigma (mm).
        ci_limit: Parameter to control the good voxel threshold. Smaller
            value relates to stricter threshold.
        debug: If true, output intermediate files to out_dir.

    Returns:
        A good voxels mask file.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Calculate coefficient of variation (cov) of the func data
        tmean_file = Path(tmp_dir).joinpath("Mean.nii.gz")
        tstd_file = Path(tmp_dir).joinpath("SD.nii.gz")
        cov_file = Path(tmp_dir).joinpath("cov.nii.gz")
        run_cmd(f"fslmaths {func_file} -Tmean {tmean_file} -odt float")
        run_cmd(f"fslmaths {func_file} -Tstd {tstd_file} -odt float")
        run_cmd(f"fslmaths {tstd_file} -div {tmean_file} {cov_file}")

        # Calculate modulated cov within cortical ribbon
        cov_ribbon_file = Path(tmp_dir).joinpath("cov_ribbon.nii.gz")
        cov_ribbon_norm_file = Path(tmp_dir).joinpath("cov_ribbon_norm.nii.gz")
        sm_norm_file = Path(tmp_dir).joinpath("SmoothNorm.nii.gz")
        cov_ribbon_norm_sm_file = Path(tmp_dir).joinpath("cov_ribbon_norm_smooth.nii.gz")
        cov_norm_modulate_file = Path(tmp_dir).joinpath("cov_norm_modulate.nii.gz")
        cov_norm_modulate_ribbon_file = Path(tmp_dir).joinpath("cov_norm_modulate_ribbon.nii.gz")
        run_cmd(f"fslmaths {cov_file} -mas {ribbon_file} {cov_ribbon_file}")
        res = run_cmd(f"fslstats {cov_ribbon_file} -M", print_output=False)
        cov_ribbon_mean = float(res.stdout)
        run_cmd(f"fslmaths {cov_ribbon_file} -div {cov_ribbon_mean} " f"{cov_ribbon_norm_file}")
        run_cmd(f"fslmaths {cov_ribbon_norm_file} -bin -s {neighborhood_smoothing} {sm_norm_file}")
        run_cmd(
            f"fslmaths {cov_ribbon_norm_file} -s {neighborhood_smoothing} "
            f"-div {sm_norm_file} -dilD {cov_ribbon_norm_sm_file}"
        )
        run_cmd(
            f"fslmaths {cov_file} -div {cov_ribbon_mean} -div "
            f"{cov_ribbon_norm_sm_file} {cov_norm_modulate_file}"
        )
        run_cmd(
            f"fslmaths {cov_norm_modulate_file} -mas {ribbon_file} {cov_norm_modulate_ribbon_file}"
        )

        # Make good voxel mask
        mask_file = Path(tmp_dir).joinpath("mask.nii.gz")
        res = run_cmd(f"fslstats {cov_norm_modulate_ribbon_file} -M", print_output=False)
        ribbon_mean = float(res.stdout)
        res = run_cmd(f"fslstats {cov_norm_modulate_ribbon_file} -S", print_output=False)
        ribbon_std = float(res.stdout)
        ribbon_upper = ribbon_mean + ribbon_std * ci_limit
        print(f"Good voxel threshold for {Path(func_file).name}: {ribbon_upper}")
        run_cmd(f"fslmaths {tmean_file} -bin {mask_file}")
        run_cmd(
            f"fslmaths {cov_norm_modulate_file} -thr {ribbon_upper} -bin -sub "
            f"{mask_file} -mul -1 -thr 1 -bin {out_file} -odt int"
        )

        # Output intermediate files if requested
        if debug:
            out_dir = Path(out_file).parent
            suffix = suffix = Path(func_file).stem.split(".")[0]
            shutil.copytree(
                tmp_dir,
                Path(out_dir).joinpath(f"temp_goodvoxel_{suffix}"),
                dirs_exist_ok=True,
            )

    return Path(out_file)


def sample_volume_to_surface(
    func_file: PathLike,
    wm_file: PathLike,
    pial_file: PathLike,
    midthickness_file: PathLike,
    out_file: PathLike,
    vol_mask_file: Optional[PathLike] = None,
    surf_mask_file: Optional[PathLike] = None,
    dilate_distance: Optional[Union[float, int]] = 10,
) -> Path:
    """Resamples data in volume space to surface space.

    Args:
        func_file: Functional image file.
        wm_file: White surface file.
        pial_file: Pial surface file.
        midthickness_file: Midthickness file.
        out_file: Output functional image file in surface space.
        vol_mask_file: Volume mask file applies to func_file. Optional.
        surf_mask_file: Surface mask file applies to sampled surface
            functional image file.
        dilate_distance: Dilate distance (mm) applies to surface sampled
            data.

    Returns:
        A functional image file in surface space.
    """

    # Sample from volume to surface
    cmd = (
        f"wb_command -disable-provenance -volume-to-surface-mapping {func_file} "
        f"{midthickness_file} {out_file} -ribbon-constrained {wm_file} {pial_file} "
    )
    if vol_mask_file is not None:
        cmd += f"-volume-roi {vol_mask_file}"
    run_cmd(cmd)

    # Dilate mapped surface
    # This step follows HCPpipeline
    # This could fix some bad vertices during the mapping
    if dilate_distance:
        run_cmd(
            f"wb_command -disable-provenance -metric-dilate {out_file} "
            f"{midthickness_file} {dilate_distance} {out_file} -nearest"
        )

    # Apply surface mask to sampled data
    if surf_mask_file is not None:
        run_cmd(
            f"wb_command -disable-provenance -metric-mask "
            f"{out_file} {surf_mask_file} {out_file}"
        )

    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)

    return Path(out_file)


def extract_func_subcortical(
    func_std_file: PathLike,
    seg_std_file: PathLike,
    template_seg_file: PathLike,
    out_file: PathLike,
    smoothing_fwhm: Optional[Union[float, int]] = None,
    debug: bool = False,
) -> Path:
    """Extracts functional data in standard subcortical regions.

    Args:
        func_std_file: Functional image file in standard volume space.
        seg_std_file: Subcortical segmentation image file in standard
            volume space.
        template_seg_file: Subcortical segmentation image file in
            standard volume space. This file is used to make a
            segmentation label file.
        out_file: Output subcortical functional image file.
        smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm). If
            None, no spatial smoothing applies to the functional data.
            Note, this operation is constrained within each subcortical
            region.
        debug: If true, output intermediate files to out_dir.

    Returns:
        A functional image file contains only subcortical region data.

    Raises:
        ValueError: func_std_file or seg_std_file is not in
            MNI152NLin6Asym space (2mm).
    """

    # Check input data in MNI space
    if ("space-MNI152NLin6Asym_res-2" not in Path(func_std_file).name) or (
        "space-MNI152NLin6Asym_res-2" not in Path(seg_std_file).name
    ):
        raise ValueError("Input data and segmentation should be in MNI152NLin6Asym space (2mm). ")

    with tempfile.TemporaryDirectory() as tmp_dir:

        # Create template subcortical dense label (in MNI152NLin6Asym space)
        roi_template_file = Path(tmp_dir).joinpath("template_roi.dlabel.nii")
        run_cmd(
            f"wb_command -disable-provenance -cifti-create-label {roi_template_file} "
            f"-volume {template_seg_file} {template_seg_file}"
        )

        # Create dense timeseries using subject subcortical ROI (in MNI152NLin6Asym space)
        func_file = Path(tmp_dir).joinpath("func.dtseries.nii")
        run_cmd(
            f"wb_command -disable-provenance -cifti-create-dense-timeseries {func_file} "
            f"-volume {func_std_file} {seg_std_file}"
        )
        # Dilate out any exact zeros voxel in func_file
        func_dil_file = Path(tmp_dir).joinpath("func_dil.dtseries.nii")
        run_cmd(
            f"wb_command -disable-provenance -cifti-dilate {func_file} "
            f"COLUMN 0 30 {func_dil_file}"
        )
        # Smoothing if requested
        if smoothing_fwhm is not None:
            sigma = convert_fwhm_to_sigma(smoothing_fwhm)
            func_dil_sm_file = Path(tmp_dir).joinpath("func_dil_sm.dtseries.nii")
            run_cmd(
                f"wb_command -disable-provenance -cifti-smoothing {func_dil_file} "
                f"0 {sigma} COLUMN {func_dil_sm_file} -fix-zeros-volume"
            )
        else:
            func_dil_sm_file = func_dil_file
        # Resample func dense timeseries to template subcortical ROI
        func_template_file = Path(tmp_dir).joinpath("func_template.dtseries.nii")
        run_cmd(
            f"wb_command -disable-provenance -cifti-resample {func_dil_sm_file} COLUMN "
            f"{roi_template_file} COLUMN ADAP_BARY_AREA CUBIC {func_template_file} "
            "-volume-predilate 10"
        )
        # Dilate again to ensure no zero in standard ROIs
        template_func_dil_file = Path(tmp_dir).joinpath("func_template_dil.dtseries.nii")
        run_cmd(
            f"wb_command -disable-provenance -cifti-dilate {func_template_file} COLUMN 0 30 "
            f"{template_func_dil_file}"
        )

        # Save functional data in subcortical to NIFTI file
        run_cmd(
            f"wb_command -disable-provenance -cifti-separate {template_func_dil_file} COLUMN "
            f"-volume-all {out_file}"
        )

        # Save intermediate files if requested
        if debug:
            out_dir = Path(out_file).parent
            suffix = Path(func_std_file).stem.split(".")[0]
            if smoothing_fwhm is not None:
                suffix += "_sm{}".format(convert_fwhm_to_str(smoothing_fwhm))
            shutil.copytree(
                tmp_dir,
                Path(out_dir).joinpath(f"temp_subcortical_{suffix}"),
                dirs_exist_ok=True,
            )

    return Path(out_file)


def make_func_map_name(
    func_file: PathLike,
    timestep: Union[float, int],
    out_file: PathLike,
    float_format: str = ":.1f",
) -> Path:
    """Writes CIFTI/GIFTI map name of a file to text file.

    This functions generates a series of map name based on the input
    repetition time, which could be used in a CIFTI dtseries file.

    Args:
        func_file: Functional image file (NIFTI or GIFTI). This file is
            used to calculate the total number of time points.
        timestep: The temporal interval of consecutive time points in
            the func_file. Usually it's the repetition time of the
            functional image.
        out_file: Output map name file.
        float_format: Float number format in the map name.

    Returns:
        A map name text file.
    """

    nvol = int(run_cmd(f"fslnvols {func_file}", print_output=False).stdout)
    mapname = ""
    for i in np.arange(0, nvol * timestep, timestep):
        mapname += ("{" + float_format + "} seconds\n").format(i)
    Path(out_file).write_text(mapname)
    return Path(out_file)


def convert_fwhm_to_sigma(fwhm: Union[float, int]) -> float:
    """Converts smoothing FWHM to gaussian sigma."""
    return float(fwhm) / (2 * np.sqrt(2 * np.log(2)))


def convert_fwhm_to_str(fwhm: Union[float, int]) -> str:
    """Converts smoothing FWHM to pure string representation."""
    return f"{fwhm:0.1f}".replace(".", "pt")
