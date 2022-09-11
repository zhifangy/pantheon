#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions relate to GLM confounds regressor."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:
#

from __future__ import annotations
from typing import Optional, Union
import os
from pathlib import Path
import numpy as np
import pandas as pd

from .regressor import make_poly_regressors
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def make_confounds_regressor(
    df: pd.DataFrame,
    out_dir: PathLike,
    demean: bool = True,
    split_into_pad_runs: bool = True,
    confounds_list: list[str] = [
        "trans_x",
        "trans_y",
        "trans_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "framewise_displacement",
    ],
    prefix: Optional[str] = None,
) -> list[Path]:
    """Makes confounds regressors file.

    This function makes 1D txt file that could be read by AFNI's
    programs. It also has the ability to make run-specific padded files.
    This is very useful when building a design matrix contains all runs.

    Args:
        df: Confounds dataframe.
        out_dir: Directory to store output file.
        demean: If true, remove mean value from each column.
        split_into_pad_runs: If true, make run-specific confounds files
            with same length as input df. Values in rows doesn't belong
            to the current run are filled with 0.
        confounds_list: Confounds names include in the output file.
            Every specified confound should present in the df.
        prefix: Filename prefix of the output file. If it's None, the
            default filename is confounds.1D (or {run_id}_confounds.1D).

    Returns:
        A confounds regressor file which could be used in AFNI's
        3dDeconvolve program.
        If 'split_into_pad_runs' is true, returning a list of filenames
        corresponds to each run in the df.

    Raises:
        ValueError: Less than 2 runs in df if 'split_into_pad_runs' is
            true.
    """

    print(f"Confounds regressor: {', '.join(confounds_list)}.")
    if prefix:
        prefix = prefix if prefix.endswith("_") else f"{prefix}_"
    else:
        prefix = ""
    # Get run list if split_into_pad_runs
    if split_into_pad_runs:
        run_list = df["run_id"].unique().tolist()
        if len(run_list) < 2:
            raise ValueError("There should be at least 2 runs if 'split_into_pad_runs' is true.")
    # Mean-center confounds regressors
    if demean:
        if split_into_pad_runs:
            confounds = (
                df.loc[:, ["run_id"] + confounds_list]
                .groupby(by=["run_id"], sort=False)
                .transform(lambda x: x - x.mean())
            )
            confounds = confounds.fillna(0)
            print("Mean center all regressors within each run.")
        else:
            confounds = (df.loc[:, confounds_list] - df.loc[:, confounds_list].mean()).fillna(0)
            print("Mean center all regressors.")
    # Or not
    else:
        confounds = df.loc[:, confounds_list].fillna(0)
    # Convert confounds regressors for per run regression
    if split_into_pad_runs:
        confounds_split = dict()
        for run_id in run_list:
            confounds_split[run_id] = np.zeros((df.shape[0], len(confounds_list)))
            confounds_split[run_id][df.run_id == run_id, :] = confounds.loc[
                df.run_id == run_id, :
            ].to_numpy()
    # Write confounds regressors to file
    fname = out_dir.joinpath(f"{prefix}confounds.1D")
    confounds_file = [fname]
    np.savetxt(fname, confounds, fmt="%.6f")
    if split_into_pad_runs:
        confounds_file = []
        for run_id in run_list:
            fname = out_dir.joinpath(f"{prefix}{run_id}_confounds.1D")
            confounds_file.append(fname)
            np.savetxt(fname, confounds_split[run_id], fmt="%.6f")

    return confounds_file


def make_good_tr_regressor(
    df: pd.DataFrame,
    out_dir: os.PathLike,
    censor_prev_tr: bool = True,
    fd_thresh: Optional[float] = 0.5,
    enorm_thresh: Optional[float] = 0.2,
    extra_censor: Optional[pd.DataFrame] = None,
    prefix: Optional[str] = None,
    dry_run: bool = False,
) -> tuple(Path, Path):
    """Calculates good TR based on motion parameters.

    Args:
        df: Confounds dataframe.
        out_dir: Directory to store output file.
        censor_pre_tr: If true, also mark the the time point before a
            bad TR as bad.
        fd_thresh: Framewise displacement threshold. TRs exceed this are
            marked as bad.
        enorm_thresh: Eucilidean norm threshold. TRs exceed this are
            marked as bad.
        extra_censor: Extra censor information. It should be a dataframe
            with a column named 'is_good'. The values in the column
            could be 1 or 0, which 1 represents good TR and 0 represents
            bad TR. This information will be combined with the fd and
            enorm based method to determine the final good TR list.
        prefix: Filename prefix of the output file. If it's None, the
            default filename is goodtr.1D and censor_info.csv.
        dry_run: If true, only print out censored TR information,
            instead of writing output files.

    Returns:
        A tuple (GoodTR, MotionCensor), where GoodTR is the filename of
        the good TR file, and MotionCensor is the filename of the
        detailed motion metric file.
    """

    if prefix:
        prefix = prefix if prefix.endswith("_") else f"{prefix}_"
    else:
        prefix = ""
    # Get run length list if ignore_first_volume_per_run
    assert "run_id" in df.columns, "Column 'run_id' not found in the input dataframe."
    run_list = df["run_id"].unique().tolist()
    run_lengths = []
    for run_id in run_list:
        run_lengths.append(df.loc[df.run_id == run_id, :].shape[0])
    # Create good tr file for timepoint censor
    good_tr = np.ones(df.shape[0], dtype=np.int16)
    motion_censor = df[[]].copy()
    # Censor TR based on Framewise displacement (L1 norm)
    if fd_thresh:
        print(f"Framewise Displacement threshold: {fd_thresh}")
        assert (
            "framewise_displacement" in df.columns
        ), "Column 'framewise_displacement' not found ..."
        fd = df["framewise_displacement"].to_numpy()
        fd = np.nan_to_num(fd, nan=0)
        # Set first volume of each run to 0
        for i in range(1, len(run_lengths)):
            fd[np.sum(run_lengths[:i])] = 0
        motion_censor["fd"] = fd
        motion_censor["fd_censor"] = np.where(motion_censor["fd"] > fd_thresh, 0, 1)
        good_tr = good_tr * motion_censor["fd_censor"].to_numpy()
    # Censor TR based on Euclidean Norm (L2 norm)
    if enorm_thresh:
        print(f"Euclidean Norm threshold: {enorm_thresh}")
        enorm = calc_motion_enorm(df)
        # Set first volume of each run to 0
        for i in range(1, len(run_lengths)):
            enorm[np.sum(run_lengths[:i])] = 0
        motion_censor["enorm"] = enorm
        motion_censor["enorm_censor"] = np.where(motion_censor["enorm"] > enorm_thresh, 0, 1)
        good_tr = good_tr * motion_censor["enorm_censor"].to_numpy()
    # Extra censor from external source
    if extra_censor:
        if "is_good" in extra_censor.columns:
            print("Calculate bad TR based on extra censor data ...")
            good_tr = good_tr * extra_censor["is_good"]
        else:
            print("Column 'is_good' not found in dataframe extra_censor.")
    # Also censor previous TR when a TR is marked as bad
    if censor_prev_tr:
        good_tr[:-1] = good_tr[:-1] * good_tr[1:]
    # Write good tr and motion censor info to file
    good_tr_file = out_dir.joinpath(f"{prefix}goodtr.1D")
    motion_censor_file = out_dir.joinpath(f"{prefix}censor_info.csv")
    if not dry_run:
        good_tr = good_tr.T.astype(np.int16)
        np.savetxt(good_tr_file, good_tr, fmt="%i")
        motion_censor.to_csv(motion_censor_file, index=False)

    n_censor = np.sum(good_tr == 0)
    pct_censor = np.mean(good_tr == 0) * 100
    print(f"Total censored TR number: {n_censor}({pct_censor:.2f}%)")

    return good_tr_file, motion_censor_file


def make_highpass_regressor(
    n_timepoint: int,
    repetition_time: Union[float, int],
    out_dir: PathLike,
    hp_freq: Union[float, int] = 0.01,
    prefix: Optional[str] = None,
) -> Path:
    """Makes highpass filter regressors.

    This function creates a set of columns of sines and cosines for the
    purpose of highpass temporal filtering. See AFNI's 1dBport for
    detailed explanations.

    Args:
        n_timepoint: Numerber of time points.
        repetition_time: Repetition time.
        out_dir: Directory to store output file.
        hp_freq: Cutoff frequency in Hz.
        prefix: Filename prefix of the output file. If it's None, the
            default filename is highpass.1D.

    Returns:
        A highpass filter regressor file which could be used in AFNI's
        3dDeconvolve program.
    """

    if prefix:
        prefix = prefix if prefix.endswith("_") else f"{prefix}_"
    else:
        prefix = ""
    out_file = Path(out_dir).joinpath(f"{prefix}highpass.1D")
    res = run_cmd(
        f"1dBport -nodata {n_timepoint} {repetition_time} -band 0 {hp_freq} -nozero",
        print_output=False,
    )
    with open(out_file, "w") as f:
        f.write(res.stdout)
    return out_file


def calc_motion_enorm(df: pd.DataFrame) -> np.ndarray:
    """Calculates euclidean norm from motion parameters.

    Args:
        df: A dataframe contains motion parameters. The column names of
            the motion parameters should be trans_x, trans_y, trans_z,
            rot_x, rot_y and rot_z.

    Returns:
        A 1d numpy array contains euclidean norm of the motion
        parameters (difference between T and T-1).
    """

    mot_par = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
    enorm = np.sqrt(
        np.sum((df[mot_par].to_numpy()[1:, :] - df[mot_par].to_numpy()[:-1, :]) ** 2, axis=1)
    )
    enorm = np.insert(enorm, 0, 0)
    return enorm


def remove_allzero_column(confounds_file: PathLike) -> tuple[Path, list]:
    """Removes all zero column in confounds regressor file.

    This functions overwrites the input confounds regressor file. It is
    useful when the head motions are very small in some direction. In
    such case, the regressors will contain only zeros under a give float
    precision, which could be problematic for GLM programs.

    Args:
        confounds_file: Confounds regressor file.

    Returns:
        A tuple (ConfoundsFile, Index), where ConfoundsFile is the
        filename of the input confounds regressor file, and the Index is
        the index of columns only have zeros.
    """

    confounds = np.loadtxt(confounds_file)
    ncol = confounds.shape[1]
    # Check each column
    sel = [True] * ncol
    for i in range(confounds.shape[1]):
        if np.allclose(confounds[:, i], 0):
            sel[i] = False
    # Remove column in place
    if np.sum(sel) != ncol:
        confounds = confounds[:, sel]
        print(f"WARNING: Removing {ncol-np.sum(sel)} all zero column!")
        np.savetxt(confounds_file, confounds, fmt="%.6f")
        # get bad column index
        allzero_column_index = list(np.arange(ncol)[np.invert(sel)])
    else:
        allzero_column_index = []
    return confounds_file, allzero_column_index


def add_polynomial(
    confounds: Union[pd.DataFrame, np.ndarray], order: int = 2
) -> Union[pd.DataFrame, np.ndarray]:
    """Add legendre polynominal regressors to confounds.

    Args:
        confounds: Confounds regressors matrix. It could be a DataFrame
            or a numpy array with samples in the 1st dimension.
        order: Largest polynomial order (degree) includes in the
            returned array.

    Returns:
        A DataFrame or numpy array contains columns of input confounds
        and legendre polynominals.
    """

    n_samples = confounds.shape[0]
    poly = make_poly_regressors(n_samples, order=order)
    if isinstance(confounds, np.ndarray):
        if confounds.ndim == 1:
            X = np.hstack((confounds[:, np.newaxis], poly))
        elif confounds.ndim == 2:
            X = np.hstack((confounds, poly))
        else:
            raise ValueError(
                "If argument 'confounds' is a numpy array, the dimension should be 1 or 2"
            )
    elif isinstance(confounds, pd.DataFrame):
        X = confounds.copy()
        for i in range(order + 1):
            X[f"poly{i}"] = poly[:, i]
    else:
        raise TypeError("Argument 'confounds' should be a DataFrame or numpy array.")

    return X
