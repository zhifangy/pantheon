#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:
#

from __future__ import annotations
from typing import Optional
from pathlib import Path
import shutil
import tempfile

from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def make_brainmask_from_atlas(atlas_file: PathLike, out_file: PathLike) -> Path:
    """Creates brainmask by refining atlas volume.

    Args:
        atlas_file: Volume brain atlas file.
        out_file: Output brain mask file.

    Returns:
        A brain mask file.
    """

    print(f"Creating brainmask from {atlas_file} ...", flush=True)
    run_cmd(f"fslmaths {atlas_file} -bin -dilD -dilD -dilD -ero -ero {out_file}")
    run_cmd(f"wb_command -disable-provenance -volume-fill-holes {out_file} {out_file}")
    run_cmd(f"fslmaths {out_file} -bin {out_file}")
    return out_file


def warp_atlas_to_reference(
    atlas_file: PathLike,
    out_file: PathLike,
    xfm_file: PathLike,
    ref_file: PathLike,
    lut_file: Optional[PathLike],
) -> Path:
    """Warps atlas file to target space.

    Args:
        atlas_file: Volume brain atlas file.
        out_file: Output brain atlas file.
        xfm_file: Spatial transformation matrix file. It should be a
            antsApplyTransforms compatible file.
        ref_file: Reference volume file for spatial transformation.
        lut_file: Lut file contains label information of the atlas_file.
            It is used to import label information to atlas NIFTI image
            header. Optional.

    Returns:
        A warpped atlas file.
    """

    # Warp atlas to MNI space
    print(f"Warpping {atlas_file} to {ref_file} ...", flush=True)
    run_cmd(
        f"antsApplyTransforms -i {atlas_file} -r {ref_file} -o {out_file} -t {xfm_file} "
        "-u int -n MultiLabel"
    )
    # Import label information to NIFTI file
    if lut_file is not None:
        run_cmd(
            f"wb_command -disable-provenance -logging SEVERE -volume-label-import "
            f"{out_file} {lut_file} {out_file} -discard-others"
        )
    return out_file


def make_cortical_ribbon(
    ref_file: PathLike,
    out_file: PathLike,
    left_wm_file: Optional[PathLike] = None,
    left_pial_file: Optional[PathLike] = None,
    right_wm_file: Optional[PathLike] = None,
    right_pial_file: Optional[PathLike] = None,
    grey_ribbon_value: int = 1,
    debug: bool = False,
) -> Path:
    """Makes cortical ribbon volume from white and pial surface.

    Args:
        ref_file: Volume image file used as reference of generated
            cortical ribbon file.
        out_file: Output cortical ribbon file.
        left_wm_file: Left white surface file.
        left_pial_file: Left pial surface file.
        right_wm_file: Right white surface file.
        right_pial_file: Right pial surface file.
        grey_ribbon_value: Index value of the ribbon voxels.
        debug: If true, output intermediate files.

    Returns:
        A cortical ribbon mask file.

    Raises:
        ValueError: No valid wm, pial surface combination is given.
    """

    # Parse input surface file
    surf = {}
    if (left_wm_file is not None) and (left_pial_file is not None):
        surf["L"] = {"wm": left_wm_file, "pial": left_pial_file}
    if (right_wm_file is not None) and (right_pial_file is not None):
        surf["R"] = {"wm": right_wm_file, "pial": right_pial_file}
    if len(surf.keys()) == 0:
        raise ValueError(
            "No valid surface combination (wm, pial) is found. Require one hemisphere at least."
        )

    ribbon = []
    with tempfile.TemporaryDirectory() as tmp_dir:

        for hemi in surf.keys():
            # Calculate distance between white and pial surface
            wm_dist_file = Path(tmp_dir).joinpath(f"hemi-{hemi}_desc-distance_wm.nii.gz")
            pial_dist_file = Path(tmp_dir).joinpath(f"hemi-{hemi}_desc-distance_pial.nii.gz")
            run_cmd(
                f"wb_command -disable-provenance -create-signed-distance-volume "
                f"{surf[hemi]['wm']} {ref_file} {wm_dist_file}"
            )
            run_cmd(
                f"wb_command -disable-provenance -create-signed-distance-volume "
                f"{surf[hemi]['pial']} {ref_file} {pial_dist_file}"
            )

            # Thresholding distance file
            wm_thr0_file = Path(tmp_dir).joinpath(f"hemi-{hemi}_desc-distance-thr0_wm.nii.gz")
            pial_thr0_file = Path(tmp_dir).joinpath(f"hemi-{hemi}_desc-distance-thr0_pial.nii.gz")
            run_cmd(f"fslmaths {wm_dist_file} -thr 0 -bin -mul 255 {wm_thr0_file}")
            run_cmd(f"fslmaths {wm_thr0_file} -bin {wm_thr0_file}")
            run_cmd(f"fslmaths {pial_dist_file} -uthr 0 -abs -bin -mul 255 {pial_thr0_file}")
            run_cmd(f"fslmaths {pial_thr0_file} -bin {pial_thr0_file}")

            # Make ribbon volume
            ribbon_file = Path(tmp_dir).joinpath(f"hemi-{hemi}_ribbon.nii.gz")
            run_cmd(f"fslmaths {pial_thr0_file} -mas {wm_thr0_file} -mul 255 {ribbon_file}")
            run_cmd(f"fslmaths {ribbon_file} -bin -mul {grey_ribbon_value} {ribbon_file} -odt int")
            ribbon.append(ribbon_file)

        # Combine ribbon from left and right hemispheres
        if len(ribbon) == 2:
            run_cmd(f"fslmaths {ribbon[0]} -add {ribbon[1]} {out_file}")
        else:
            shutil.copy(ribbon[0], out_file)

        # Output intermediate files if requested
        if debug:
            out_dir = Path(out_file).parent
            suffix = Path(ref_file).stem.split(".")[0]
            shutil.copytree(
                tmp_dir,
                Path(out_dir).joinpath(f"temp_cortical_ribbon_{suffix}"),
                dirs_exist_ok=True,
            )

    return out_file
