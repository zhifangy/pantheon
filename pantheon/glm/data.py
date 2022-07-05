#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to data used in GLM."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Optional, Union
from pathlib import Path
import tempfile

from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def scale_func_image(
    in_file: PathLike,
    out_file: PathLike,
    mask_file: Optional[PathLike] = None,
    cifti: bool = False,
) -> Path:
    """Scales a image in range of 0 to 200.

    By scaling BOLD data in range [0, 200] with mean equals to 100, the
    beta derived from GLM roughly represents the percentage of signal
    change (all regressors are scaled to unit size). See AFNI documents
    for details.

    Args:
        in_file: A functional image file.
        out_file: Output functional file.
        mask_file: A mask file applies to input image during scaling. It
            only enables when input file is a NIFTI or GIFTI file.
        cifti: If true, treat input file as a CIFTI file and scale it
            with wb_command.

    Returns:
        A functional image file.
    """

    with tempfile.TemporaryDirectory() as tmp_dir:

        if cifti:
            # Calculate temporal mean image
            mean_file = Path(tmp_dir, "mean.dscalar.nii")
            run_cmd(f"wb_command -disable-provenance -cifti-reduce {in_file} MEAN {mean_file}")
            # Scale func image in to 0-200 range
            scale_expr = f"min(200,(a/b)*100)*(a>0)*(b>0)"
            run_cmd(
                f"wb_command -disable-provenance -cifti-math '{scale_expr}' "
                f"{out_file} -var a {in_file} -var b {mean_file} -select 1 1 -repeat",
                print_output=False,
            )
        else:
            # Calculate temporal mean image
            if Path(in_file).suffix.endswith("gii"):
                mean_file = Path(tmp_dir, "mean.shape.gii")
            else:
                mean_file = Path(tmp_dir, "mean.nii.gz")
            run_cmd(f"3dTstat -mean -prefix {mean_file} {in_file}")
            # Scale func image in to 0-200 range
            if mask_file:
                scale_expr = f"c*min(200,(a/b)*100)*step(a)*step(b)"
                run_cmd(
                    f"3dcalc -a {in_file} -b {mean_file} -c {mask_file} "
                    f"-expr '{scale_expr}' -prefix {out_file}"
                )
            else:
                scale_expr = f"min(200,(a/b)*100)*step(a)*step(b)"
                run_cmd(
                    f"3dcalc -a {in_file} -b {mean_file} "
                    f"-expr '{scale_expr}' -prefix {out_file}"
                )

    return Path(out_file)


def calc_run_length(in_file: Union[PathLike, list[PathLike]], cifti: bool = False) -> list[int]:
    """
    Calculates functional image length (number of time points).

    Args:
        in_file: A single or a list of functional image files.
        cifti: If true, treat input file as a CIFTI file.

    Returns:
        A list of lengths (number of timepoints) of the input files.
    """

    if not isinstance(in_file, list):
        in_file = [in_file]
    run_length = []
    for f in in_file:
        # CIFTI or GIFTI file
        if cifti or (Path(f).suffix.endswith("gii")):
            cmd = ["wb_command", "-file-information", f, "-only-number-of-maps"]
        # NIFTI file
        else:
            cmd = ["fslnvols", f]
        ntp = int(run_cmd(cmd, print_output=False).stdout)
        run_length.append(ntp)
    return run_length
