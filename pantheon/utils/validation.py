#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Argument validation functions"""

from __future__ import annotations
from typing import Optional, Union, Literal

from .typing import PathLike


def check_file_list(file_list: list[PathLike], n: Optional[int] = None) -> int:
    """Checks list of filenames.

    Makes sure each filename in the file_list is a valid file on disk.
    Optionally, checks the number of filenames meet expection.

    Args:
        file_list: A list of filenames.
        n: Expected number of filenames in file_list.

    Returns:
        Number of filenames in the file_list.

    Raises:
        ValueError: Number of filenames in the file_list doesn't meet
            expectation.
        FileNotFoundError: One or multiple files in the file_list is not
            found.
    """

    n_file = len(file_list)
    if n_file == 0:
        raise ValueError("Argument file_list is empty.")
    if n and n_file != n:
        raise ValueError(f"Number of files in the list is {n_file}. Expecting {n}.")
    for f in file_list:
        if not f.is_file():
            raise FileNotFoundError(f"File {f} is not found.")

    return n_file


####################
# Conform argument #
####################


def conform_sub_id(sub_id: Union[int, str], with_prefix: bool = False) -> str:
    """Conforms Subject ID.

    Args:
        sub_id: Subject ID
        with_prefix: If true, the conformed Subject ID always starts
            with 'sub-'.

    Returns:
        Conformed Subject ID.
        If input is a int, conform it as zero padded string (e.g., 001).

    Raises:
        TypeError: Input sub_id is not int or string type.
    """

    if not isinstance(sub_id, (int, str)):
        raise TypeError("Argument sub_id should be a int or str.")
    if isinstance(sub_id, int):
        sub_id = f"{int(sub_id):03d}"
    if (sub_id.startswith("sub-")) and (not with_prefix):
        sub_id = sub_id[4:]
    if (not sub_id.startswith("sub-")) and with_prefix:
        sub_id = f"sub-{sub_id}"
    return sub_id


def conform_run_id(run_id: Union[int, str], with_prefix: bool = False) -> str:
    """Conforms Subject ID.

    Args:
        run_id: Run ID
        with_prefix: If true, the conformed Run ID always starts
            with 'run-'.

    Returns:
        Conformed Run ID.

    Raises:
        TypeError: Input run_id is not int or string type.
    """

    if not isinstance(run_id, (int, str)):
        raise TypeError("Argument run_id should be a int or str.")
    run_id = str(run_id)
    if (run_id.startswith("run-")) and (not with_prefix):
        run_id = run_id[4:]
    if (not run_id.startswith("run-")) and with_prefix:
        run_id = f"run-{run_id}"
    return run_id


def conform_task_id(task_id: str, with_prefix: bool = False) -> str:
    """Conforms Task ID.

    Args:
        task_id: Task ID
        with_prefix: If true, the conformed Task ID always starts
            with 'task-'.

    Returns:
        Conformed Task ID.

    Raises:
        TypeError: Input task_id is not a string.
    """

    if not isinstance(task_id, str):
        raise TypeError("Argument task_id should be a string.")
    if (task_id.startswith("task-")) and (not with_prefix):
        task_id = task_id[5:]
    if (not task_id.startswith("task-")) and with_prefix:
        task_id = f"task-{task_id}"
    return task_id


############################
# Parse and check argument #
############################


def parse_space(space: str, valid_list: list[str] = ["fsnative", "fsLR"]) -> str:
    """Parses and validates argument 'space'.

    Args:
        space: Spatial space.
        valid_list: Valid values of space.

    Returns:
        A valid space.

    Raises:
        ValueError: space value is invalid.
    """

    if space not in valid_list:
        raise ValueError(f"Invalid space: {space}. Valid: {', '.join(valid_list)}.")
    return space


def parse_hemi(
    hemi: str,
    valid_list: list[str] = ["L", "R"],
    structure_format: Literal["cifti", "gifti"] = "cifti",
) -> tuple[str, str]:
    """Parses and validates argument 'hemi'.

    Args:
        hemi: Brain hemisphere.
        valid_list: Valid values of hemi.
        structure_format: Surface structure name format. For example,
            CORTEX_LEFT. Valid: cifti, gifti.

    Returns:
        A tuple (hemi, structure), where hemi indicates left or right
        hemisphere, and structure is the structure name of the surface.
        For example, CORTEX_LEFT (cifti format).

    Raises:
        ValueError: hemi value is invalid.
    """

    if hemi not in valid_list:
        raise ValueError(f"Invalid hemi: {hemi}. Valid: {', '.join(valid_list)}.")
    struc_name = {
        "L": {"cifti": "CORTEX_LEFT", "gifti": "CortexLeft"},
        "R": {"cifti": "CORTEX_RIGHT", "gifti": "CortexRight"},
    }
    structure = struc_name[hemi][structure_format]
    return hemi, structure


def parse_mesh_density(
    mesh_den: str, valid_list: list[str] = ["fsnative", "164k", "59k", "32k"]
) -> str:
    """Parses and validates argument 'mesh_den'.

    Args:
        mesh_den: Surface mesh density.
        valid_list: Valid values of mesh_den.

    Returns:
        A valid mesh density string.

    Raises:
        ValueError: mesh_den value is invalid.
    """

    if mesh_den not in valid_list:
        raise ValueError(f"Invalid mesh_den: {mesh_den}. Valid: {', '.join(valid_list)}.")
    return mesh_den


def parse_registration_method(
    registration_method: str, valid_list: list[str] = ["FS", "MSMSulc"]
) -> str:
    """Parses and validates argument 'registration_method'.

    Args:
        registration_method: Spatial registration method.
        valid_list: Valid values of registration_method.

    Returns:
        A valid registration method string.

    Raises:
        ValueError: registration_method value is invalid.
    """

    if registration_method not in valid_list:
        raise ValueError(
            f"Invalid registration_method: {registration_method}. Valid: {', '.join(valid_list)}."
        )
    return registration_method


def parse_smoothing_fwhm(
    smoothing_fwhm: Optional[list[int, float]] = None, remove_zero: bool = False
) -> Optional[list[float]]:
    """Parses and validates argument 'smoothing_fwhm'.

    Args:
        smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).
            It could be a number, a list of numbers or None.
        remove_zero: If true, remove 0 from the smoothing_fwhm list.

    Returns:
        A list of valid smoothing FWHM value. It could be None if the
        input is None.

    Raises:
        ValueError: smoothing_fwhm value is invalid.
    """

    if smoothing_fwhm is None:
        return None
    if isinstance(smoothing_fwhm, (int, float)):
        smoothing_fwhm = [smoothing_fwhm]
    elif isinstance(smoothing_fwhm, list):
        for fwhm in smoothing_fwhm:
            if not isinstance(fwhm, (int, float)):
                raise ValueError("Elements in smoothing_fwhm should be a float or int.")
    else:
        raise ValueError("Argument smoothing_fwhm should be a number of a list of numbers.")

    conform_fwhm = []
    for fwhm in smoothing_fwhm:
        # remove 0 if requested
        if (fwhm != 0) or ((fwhm == 0) and (not remove_zero)):
            conform_fwhm.append(float(fwhm))
    if len(conform_fwhm) == 0:
        conform_fwhm = None
    return conform_fwhm


def parse_roi_id(roi_id: str) -> tuple[str, str]:
    """Parses and validates argument 'roi_id'.

    Args:
        roi_id: ROI ID. It should be in the format like 'NAME-HEMI'.
            For example, AG-L represents AG in left hemisphere. If '-L'
            or '-R' is omitted, it represents bilateral ROI.

    Returns:
        A tuple (ROI name, hemi), where ROI name indicates the name of
        the ROI without hemisphere suffix, and hemi indicates left or
        right brain hemisphere. If it's a bilateral ROI, hemi is LR.
    Raises:
        ValueError: roi_id format is incorrect.
    """

    parts = roi_id.split("-")
    cond1 = (parts[-1] == "L") or (parts[-1] == "R")  # ending with L or R
    cond2 = len(parts) == 1
    cond3 = len(parts) > 2
    # Ensure roi_id ends with 'L', 'R' with exactly one '-'. Or it
    # doesn't contain any '-' (indicates bilateral or midline ROI)
    if (not (cond1 or cond2)) or cond3:
        raise ValueError(
            "Argument 'roi_id' should be a string with one and at most one '-'.\n"
            "Before the dash is the ROI name and after dash is the hemisphere (L or R).\n"
            "If this is a bilateral or midline ROI, the hemisphere should be omitted."
        )
    roi_name = parts[0]
    hemi = parts[1] if len(parts) == 2 else "LR"
    return roi_name, hemi
