#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Union, Literal
from pathlib import Path
import tempfile

from .bold import convert_fwhm_to_sigma
from ..image.gifti import sanitize_gii_metadata
from ..utils.validation import parse_hemi
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def make_midthickness_surface(
    hemi: Literal["L", "R"], wm_file: PathLike, pial_file: PathLike, out_file: PathLike
) -> Path:
    """Creates midthickness surface.

    Args:
        hemi: Brain hemisphere.
        wm_file: White surface file.
        pial_file: Pial surface file.
        out_file: Output midthickness surface file.

    Returns:inflated_out_file
        Midthickness surface file.
    """

    # Parse hemisphere
    hemi, structure = parse_hemi(hemi)
    # Create midthickness by averaging white and pial surface
    print(f"Creating midthickness surface: {Path(out_file).name} ...", flush=True)
    run_cmd(
        f"wb_command -disable-provenance -surface-average {out_file} "
        f"-surf {wm_file} -surf {pial_file}"
    )
    # Set GIFTI metadata
    run_cmd(
        f"wb_command -disable-provenance -set-structure {out_file} "
        f"{structure} -surface-type ANATOMICAL -surface-secondary-type MIDTHICKNESS"
    )
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file, da_meta={"Name": ""})
    return Path(out_file)


def make_inflated_surface(
    midthickness_file: PathLike,
    inflated_out_file: PathLike,
    very_inflated_file: PathLike,
    inflate_extra_scale: Union[float, int] = 1.0,
) -> list[Path]:
    """Creates inflated and very inflated surfaces.

    Args:
        midthickness_file: Midthickness surface file.
        inflated_out_file: Output inflated surface file.
        very_inflated_file: Output veryinflated surface file.
        inflate_extra_scale: Extra iteration scaling value. This value
            is used in function calc_inflation_scale to calculate the
            final iteration scaling value.

    Returns:
        A tuple (inflated, veryinflated), where inflated is the inflated
        surface file and veryinflated is the veryinflated surface file.
    """

    inflation_scale = calc_inflation_scale(
        midthickness_file, inflate_extra_scale=inflate_extra_scale
    )
    print(f"Creating inflated surfaces: {inflated_out_file} ...", flush=True)
    print(f"Inflation scale: {inflation_scale}", flush=True)
    run_cmd(
        "wb_command -disable-provenance -surface-generate-inflated "
        f"{midthickness_file} {inflated_out_file} {very_inflated_file} "
        f"-iterations-scale {inflation_scale}"
    )
    # Cleanup metadata
    _ = sanitize_gii_metadata(inflated_out_file, inflated_out_file, da_meta={"Name": ""})
    _ = sanitize_gii_metadata(very_inflated_file, very_inflated_file, da_meta={"Name": ""})
    return Path(inflated_out_file), Path(very_inflated_file)


def make_nomedialwall_roi(
    thickness_file: PathLike,
    midthickness_file: PathLike,
    out_file: PathLike,
    gifti_map_name: str = "nomedialwall",
) -> Path:
    """Makes (no)medialwall ROI by thresholding the surface thickness.

    Args:
        thickness_file: Surface thickness metric file.
        midthickness_file: Midthickness surface file.
        out_file: Output nomedialwall mask file.
        gifti_map_name: GIFTI map name of the out_file.

    Returns:
        Nomedialwall mask file.
    """

    # (No)medialwall region defined as vertexs with abs(thickness) > 0
    print(
        f"Creating (no)medialwall ROI from surface thickness: {out_file} ...",
        flush=True,
    )
    run_cmd(
        f"wb_command -disable-provenance -metric-math 'thickness > 0' {out_file} "
        f"-var thickness {thickness_file}"
    )
    run_cmd(
        f"wb_command -disable-provenance -metric-fill-holes {midthickness_file} "
        f"{out_file} {out_file}"
    )
    run_cmd(
        f"wb_command -disable-provenance -metric-remove-islands {midthickness_file} "
        f"{out_file} {out_file}"
    )
    # Set GIFTI metadata
    run_cmd(f"wb_command -disable-provenance -set-map-names {out_file} -map 1 {gifti_map_name}")
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)
    return Path(out_file)


def refine_nomedialwall_roi(
    roi_file: PathLike,
    wrapped_sphere_file: PathLike,
    template_roi_file: PathLike,
    template_sphere_file: PathLike,
    out_file: PathLike,
) -> Path:
    """Refines native (no)medialwall ROI using template ROI.

    Args:
        roi_file: (no)medialwall ROI in native space with native mesh.
        wrapped_sphere_file: Surface sphere file warpped in template
            space with native mesh.
        template_roi_file: Template (no)medialwall ROI file.
        template_sphere_file: Template surface sphere file.
        out_file: Output (no)medialwall ROI file.

    Returns:
        Refined (no)medialwall ROI file.
    """

    print(f"Refining (no)medialwall ROI: {roi_file} ...", flush=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Resample template ROI to native space
        resampled_roi_file = Path(tmp_dir).joinpath(
            "ResampledTemplate_desc-nomedialwall_probseg.shape.gii"
        )
        run_cmd(
            f"wb_command -disable-provenance -metric-resample {template_roi_file} "
            f"{template_sphere_file} {wrapped_sphere_file} BARYCENTRIC "
            f"{resampled_roi_file} -largest"
        )
        # Combine native and template ROI (mostly add regions near the hippocampus)
        run_cmd(
            "wb_command -disable-provenance -metric-math '(native + template) > 0' "
            f"{out_file} -var native {roi_file} -var template {resampled_roi_file}"
        )
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)
    return Path(out_file)


def mask_metric_nomedialwall(
    metric_file: PathLike,
    roi_file: PathLike,
    midthickness_file: PathLike,
    out_file: PathLike,
) -> Path:
    """Applies (no)medialwall mask to surface metric.

    Args:
        metric_file: Surface metric file.
        roi_file: Nomedialwall ROI file.
        midthickness_file: Midthickness surface file.
        out_file: Output surface metric file.

    Returns:
        Masked surface metric file.
    """

    print(f"Masking metric file: {metric_file} ...", flush=True)
    # Dilate metric by 10mm
    run_cmd(
        f"wb_command -disable-provenance -metric-dilate {metric_file} "
        f"{midthickness_file} 10 {out_file} -nearest"
    )
    # Apply (no)medialwall ROI
    run_cmd(f"wb_command -disable-provenance -metric-mask {out_file} {roi_file} {out_file}")
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)
    return Path(out_file)


def smooth_metric(
    metric_file: PathLike,
    midthickness_file: PathLike,
    out_file: PathLike,
    smoothing_fwhm: float,
    roi_file: Optional[PathLike] = None,
) -> Path:
    """Smooths surface metric.

    Args:
        metric_file: Surface metric file.
        midthickness_file: Midthickness surface file.
        out_file: Output surface metric file.
        smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).
        roi_file: Surface mask file. Optional. If it is given, smoothing
            is constrained within that mask.

    Returns:
        Smoothed surface metric file.
    """

    # Parse smoothing fwhm
    sigma = convert_fwhm_to_sigma(smoothing_fwhm)
    # Smoothing
    cmd = (
        "wb_command -disable-provenance -metric-smoothing "
        f"{midthickness_file} {metric_file} {sigma} {out_file} "
    )
    if roi_file is not None:
        cmd += f"-roi {roi_file}"
    run_cmd(cmd)
    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)
    return Path(out_file)


def calc_inflation_scale(in_file: PathLike, inflate_extra_scale: Union[float, int] = 1.0) -> float:
    """Calculates surface inflation scale factor.

    Args:
        in_file: Surface file used for creating inflated surfaces.
        inflate_extra_scale: Extra scaling value multiply to the value
            calculated from the number of vertices.
    Returns:
        Inflation scaling value used in program
        'wb_command -surface-generate-inflated'
    """

    # This formula is from HCPpipeline
    # https://github.com/Washington-University/HCPpipelines/blob/1334b35ab863540044333bbdec70a68fb19ab611/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L337
    # Find vertex number
    info = run_cmd(f"wb_command -file-information {in_file}", print_output=False).stdout
    for i in info.split("\n"):
        if "Number of Vertices" in i:
            num_vertex = int(i.split(":")[1])
    # Calulate final scale
    inflation_scale = inflate_extra_scale * 0.75 * num_vertex / 32492
    return inflation_scale
