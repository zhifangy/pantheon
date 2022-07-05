#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to GLM regressor."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
import numpy as np


def make_run_start_string(run_length: list[int]) -> str:
    """Generates run start index string.

    This function makes a string contains the TR index of the start of
    each run. It is used for the '-concat' option in AFNI's 3dDeconvolve
    program.

    Args:
        run_length: A list of number indicates the length (number of TR)
            of each run.

    Returns:
        A string contains run start index which could be used in AFNI's
        3dDeconvolve program.
    """

    run_start = np.cumsum(np.insert(run_length, 0, 0), dtype=np.int16)[:-1]
    run_start = "1D: " + " ".join([str(i) for i in run_start])
    return run_start
