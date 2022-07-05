#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Project main workhorse."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:

from __future__ import annotations
from typing import Optional, Union, Literal, Any
from pathlib import Path
import warnings
import yaml
import re
import numpy as np
import pandas as pd
import nibabel as nib

from .masking.roi import parse_roi_id, make_roi_from_spec, unmask
from .image.cifti import read_dscalar, read_dtseries, read_dscalar_roi, read_dtseries_roi
from .utils.validation import (
    check_file_list,
    conform_sub_id,
    conform_run_id,
    conform_task_id,
    parse_hemi,
)
from .utils.typing import PathLike, StatMapView, SurfaceView


class Layout:
    """Project file layout class.

    This class provides a series of functions to get most useful files
    under root project directory.

    Attributes:
        bids_subject_list: All SubjectID in the BIDS dataset.
        subject_list: SubjectID after data exclusion.
        task_list: All TaskID in the BIDS dataset.
        task_info: A dict contains information of the task. Usually it
            contains information of data exclusion.
        func_regex: A regular expression to match a task related file.
    """

    def __init__(self, project_root_dir: PathLike) -> None:
        """Initializes class.

        Args:
            project_root_dir: Root directory of the project.
                All subdirectories are relative to this directory.
        """

        #############
        # Directories
        #############
        self.base_dir = Path(project_root_dir)
        self.code_dir = self.base_dir.joinpath("code", "pantheon")
        self.bids_dir = self.base_dir.joinpath("data", "bidsdata")
        self.deriv_dir = self.base_dir.joinpath("data", "derivatives")
        self.srcdata_dir = self.base_dir.joinpath("data", "sourcedata")
        self.extdata_dir = self.base_dir.joinpath("data", "external")
        self.metadata_dir = self.base_dir.joinpath("data", "metadata")
        self.tmp_dir = self.base_dir.joinpath("temporary_files")
        self.fs_dir = self.deriv_dir.joinpath("freesurfer")
        self.fmriprep_dir = self.deriv_dir.joinpath("fmriprep")
        self.preproc_dir = self.deriv_dir.joinpath("preprocessed")
        self.roi_dir = self.deriv_dir.joinpath("roi")
        self.singletrial_dir = self.deriv_dir.joinpath("singletrial_response")

        #############################
        # Project specific parameters
        #############################
        # Subject list from BIDS dataset
        self.bids_subject_file = self.bids_dir.joinpath("participants.tsv")
        self.bids_subject_list = list(
            pd.read_csv(self.bids_subject_file, sep="\t")["participant_id"]
        )
        # Data validation infomation
        # task name, default run list
        # subject exclusion, run exclusion
        self.validataion_file = self.metadata_dir.joinpath("data_validation.yaml")
        if self.validataion_file.is_file():
            with open(self.validataion_file, "r") as f:
                self.dv = yaml.load(f, Loader=yaml.CLoader)
            # Subject list (after exclusion)
            self.subject_list = [
                i for i in self.bids_subject_list if i not in self.dv["exclude_subject"]
            ]
            # Task information
            self.task_info = self.dv["task"]
            self.task_list = list(self.dv["task"].keys())
        else:
            warnings.warn(
                f"\nData validation file {self.validataion_file} not found.\n"
                "Use default bids_subject_list.\n"
                "Task infomation not available."
            )
        # Regular expression to match task files
        self.func_regex = (
            r"(?P<sub_id>sub-[^\W_]+)_(?P<ses_id>ses-[^\W_]+)?_?"
            + r"(?P<task_id>task-[^\W_]+)_(?P<run_id>run-\d+)?_?"
        )

    ################
    # Raw data files
    ################

    # BIDS file
    def get_anat_file(
        self,
        sub_id: Union[int, str],
        suffix: str = "T1w",
        ses_id: Optional[str] = None,
        modifier: Optional[str] = None,
    ) -> list[Path]:
        """Gets anat file in BIDS directory.

        Args:
            sub_id: SubjectID.
            suffix: BIDS anatomical file suffix (e.g., T1w).
            ses_id: SessionID. Optional.
            modifier: Any possible filename modifier after session and
                before suffix part.

        Returns:
            A list of anatomical file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        modifier = f"{modifier}*" if modifier else ""
        file_list = sorted(
            self.bids_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}*{modifier}_{suffix}.nii.gz"
            )
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_func_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[Union[str, int]]]] = None,
        ses_id: Optional[str] = None,
        suffix: str = "bold",
        modifier: Optional[str] = None,
        exclude: bool = False,
    ) -> list[Path]:
        """Gets func file in BIDS directory.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            suffix: Filename suffix after modifier and before extexsion.
            modifier: Any possible filename modifier after session and
                before suffix part.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of functional file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        modifier = f"{modifier}*" if modifier else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.bids_dir.joinpath(sub_id).glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}*{modifier}_{suffix}.nii.gz"
                )
            )
        check_file_list(file_list, n=len(run_list))
        return file_list

    def get_beh_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        suffix: str = "events",
        modifier: Optional[str] = None,
        exclude: bool = False,
    ) -> list[Path]:
        """Gets behavior file in BIDS directory.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            suffix: Filename suffix after modifier and before extexsion.
            modifier: Any possible filename modifier after session and
                before suffix part.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of behavior file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        modifier = f"{modifier}*" if modifier else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.bids_dir.joinpath(sub_id).glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}*{modifier}_{suffix}.tsv"
                )
            )
        check_file_list(file_list, n=len(run_list))
        return file_list

    ####################
    # Preprocessed files
    ####################

    def get_fmriprep_anat_file(
        self,
        sub_id: Union[int, str],
        ses_id: Optional[str] = None,
        space: Optional[str] = None,
        suffix: Literal["T1w", "mask", "dseg", "probseg"] = "T1w",
        modifier: Optional[str] = None,
    ) -> list[Path]:
        """Gets fMRIprep output anat file.

        Args:
            sub_id: SubjectID.
            ses_id: SessionID. Optional.
            space: Image space.If None, selecting T1w space as default.
            suffix: Filename suffix. Default is T1w, which selecting the
                preprocessed T1w file. Other options are mask, dseg, and
                probseg. For dseg and probseg, the modifier is usuarlly
                required.
            modifier: Any possible filename modifier after space and
                before suffix part.

        Returns:
            A list of fMRIprep preprocessed anatomical file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        space = f"_space-{space}" if space else ""
        modifier = f"{modifier}*" if modifier else ""
        file_list = sorted(
            self.fmriprep_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}{space}*{modifier}_{suffix}.nii.gz"
            )
        )
        # filter out files of other spaces if requested is T1w
        # the space part in the filename is omitted for T1w space
        if not space:
            file_list = [f for f in file_list if "space-" not in f.name]
        check_file_list(file_list, n=1)
        return file_list

    def get_fmriprep_anat_xfm_file(
        self,
        sub_id: Union[int, str],
        src_space: str = "T1w",
        trg_space: str = "MNI152NLin2009cAsym",
        ses_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> list[Path]:
        """Gets fMRIprep output anat spatial transformation file.

        Args:
            sub_id: SubjectID.
            src_space: Source space of the spatial transformation.
            trg_space: Target space of the spatial transformation.
            ses_id: SessionID. Optional.
            run_id: RunID. Optional.

        Returns:
            A list of fMRIPrep generated spatial transformation file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        run_id = f"_run-{run_id}" if run_id else ""
        file_list = sorted(
            self.fmriprep_dir.joinpath(sub_id).glob(
                f"**/{sub_id}{ses_id}{run_id}_from-{src_space}_to-{trg_space}_mode-image_xfm.*"
            )
        )
        check_file_list(file_list, n=1)
        return file_list

    def get_fmriprep_func_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        space: str = "T1w",
        suffix: Literal["bold", "boldref", "mask", "dseg"] = "bold",
        modifier: Optional[str] = None,
        exclude: bool = False,
    ) -> list[Path]:
        """Gets fMRIPrep output func file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            space: Image space.
            suffix: If 'bold', selecting preprocessed func file. If
                'boldref', selecting reference BOLD file. If 'mask',
                selecting run-specific brain mask. If 'dseg', selecting
                FreeSurfer generated segmentation.
            modifier: Any possible filename modifier after space and
                before suffix part.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of fMRIprep preprocessed functional file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        space = f"_space-{space}" if space != "" else space
        modifier = f"{modifier}*" if modifier else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.fmriprep_dir.joinpath(sub_id).glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}{space}*{modifier}_{suffix}.nii.gz"
                )
            )
        check_file_list(file_list, n=len(run_list))
        return file_list

    def get_confound_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        exclude: bool = False,
    ) -> list[Path]:
        """Gets fMRIPrep output confound regressor file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of fMRIPrep generated confound regressor file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.fmriprep_dir.joinpath(sub_id).glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}_desc-confounds_timeseries.tsv"
                )
            )
        check_file_list(file_list, n=len(run_list))
        return file_list

    def get_preproc_surf_file(
        self,
        sub_id: Union[int, str],
        hemi: Literal["L", "R"],
        surf_id: str,
        ses_id: Optional[str] = None,
        space: str = "fsLR",
        mesh_den: str = "32k",
        desc: Optional[str] = "MSMSulc",
    ) -> list[Path]:
        """Gets preprocessed surface file.

        Args:
            sub_id: SubjectID.
            hemi: Brain hemisphere. Valid: L, R.
            surf_id: Surface name. E.g., pial, sphere, probseg.
            ses_id: SessionID. Optional.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.

        Returns:
            A list of preprocessed surface file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        hemi, _ = parse_hemi(hemi)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        desc = f"{desc}*" if desc else ""
        file_list = sorted(
            self.preproc_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}_hemi-{hemi}_"
                f"space-{space}_den-{mesh_den}*{desc}_{surf_id}.*.gii"
            )
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_preproc_nomedialwall_roi_file(
        self,
        sub_id: Union[int, str],
        hemi: Literal["L", "R"],
        ses_id: Optional[str] = None,
    ) -> list[Path]:
        """Gets native space nomedialwall ROI file.

        Args:
            sub_id: SubjectID.
            hemi: Brain hemisphere. Valid: L, R.
            surf_id: Surface file name. E.g., pial, sphere, probseg.
            ses_id: SessionID. Optional.

        Returns:
            A list of native space nomedialwall ROI file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        hemi, _ = parse_hemi(hemi)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        file_list = sorted(
            self.preproc_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}_hemi-{hemi}_space-fsnative_"
                f"den-fsnative_desc-nomedialwall_probseg.shape.gii"
            )
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_preproc_subcortical_roi_file(
        self, sub_id: Union[int, str], ses_id: Optional[str] = None, space: str = "MNI152NLin6Asym"
    ) -> list[Path]:
        """Gets subcortical ROI file.

        Args:
            sub_id: SubjectID.
            ses_id: SessionID. Optional.
            space: Image space.

        Returns:
            A list of subcortical ROI file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        file_list = sorted(
            self.preproc_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}_space-{space}*_desc-Subcortical_dseg.nii.gz"
            )
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_preproc_anat_cifti_file(
        self,
        sub_id: Union[int, str],
        metric: str,
        ses_id: Optional[str] = None,
        space: str = "fsLR",
        mesh_den: str = "32k",
        desc: Optional[str] = "MSMSulc",
    ) -> list[Path]:
        """Gets preprocessed surface metric CIFTI file.

        Args:
            sub_id: SubjectID.
            metric: Surface metric name. E.g., curv, distortion.
            ses_id: SessionID. Optional.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.

        Returns:
            A list of preprocessed surface metric CIFTI file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        desc = f"{desc}*" if desc else ""
        file_list = sorted(
            self.preproc_dir.joinpath(sub_id).glob(
                f"**/anat/{sub_id}{ses_id}_space-{space}_den-{mesh_den}*{desc}_{metric}.*.nii"
            )
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_preproc_func_cifti_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "bold",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
    ) -> list[Path]:
        """Gets preprocessed functional CIFTI file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., bold, mean.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of preprocessed functional CIFTI file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        desc = f"{desc}*" if desc else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.preproc_dir.joinpath(sub_id).glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}_"
                    f"space-{space}_den-{den}*{desc}_{metric}.*.nii"
                )
            )
        _ = check_file_list(file_list, n=len(run_list))
        return file_list

    ##############################
    # Singletrial estimation files
    ##############################

    def get_singletrial_response_cifti_file(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "beta",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
    ) -> list[Path]:
        """Gets singletrial response CIFTI file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., beta, tstat.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of singletrial response CIFTI file.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        run_list = self.get_run_id(sub_id, task_id, run_id, exclude=exclude)
        if not ((len(run_list) == 1) & (run_list[0] == "")):
            run_list = [f"_{i}" for i in run_list]
        ses_id = f"_ses-{ses_id}" if ses_id else ""
        desc = f"_desc-{desc}" if desc else ""
        file_list = []
        for run_id in run_list:
            file_list += sorted(
                self.singletrial_dir.joinpath(f"{sub_id}").glob(
                    f"**/{sub_id}{ses_id}_{task_id}{run_id}_"
                    f"space-{space}_den-{den}{desc}_{metric}.dscalar.nii"
                )
            )
        _ = check_file_list(file_list, n=len(run_list))
        return file_list

    ################
    # Metadata files
    ################

    def get_metadata_file(self, name: str, ext: str = "yaml") -> list[Path]:
        """Gets metadata file.

        This function gets a metadate file under
        ${project_root_dir}/data/metadata directory.

        Returns:
            A list of metadata file.
        """

        file_list = [self.metadata_dir.joinpath(f"{name}.{ext}")]
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_template_info_file(self) -> list[Path]:
        """Gets brain template infomation file.

        This function gets the template.yaml under
        ${project_root_dir}/data/metadata directory.

        Returns:
            A list of brain template information file.
        """
        return self.get_metadata_file("template")

    def get_atlas_info_file(self) -> list[Path]:
        """Gets brain atlas infomation file.

        This function gets the atlas.yaml under
        ${project_root_dir}/data/metadata directory.

        Returns:
            A list of the atlas information file.
        """
        return self.get_metadata_file("atlas")

    def get_roi_definition_file(self) -> list[Path]:
        """Gets default ROI definition file.

        This function gets the roi_definition.yaml under
        ${project_root_dir}/data/metadata directory.

        Returns:
            A list of the ROI definition file.
        """
        return self.get_metadata_file("roi_definition")

    def get_roi_list_file(self) -> list[Path]:
        """Gets ROI list file.

        This function gets the roi_list.yaml under
        ${project_root_dir}/data/metadata directory.

        Returns:
            A list of the ROI list file.
        """
        return self.get_metadata_file("roi_list")

    ################
    # External files
    ################

    def get_std_surf_file(
        self, template_name: str, hemi: Literal["L", "R"], surf_id: str, desc: Optional[str] = None
    ) -> list[Path]:
        """
        Gets template surface file.

        Args:
            template_name: Template name. The name shoule be defined in
                metadata file template.yaml.
            hemi: Brain hemisphere. Valid: L, R.
            surf_id: Surface name. E.g., pial, sphere.
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.

        Returns:
            A list of template surface file.
        """

        with open(self.get_template_info_file()[0], "r") as f:
            template_info = yaml.load(f, Loader=yaml.CLoader)
        if template_name not in template_info.keys():
            raise ValueError(f"Template {template_name} is not defined in the template.yaml file.")
        temp_dir = template_info[template_name]
        desc = f"{desc}*" if desc else ""
        file_list = sorted(
            self.base_dir.joinpath(temp_dir).glob(f"*hemi-{hemi}*{desc}_{surf_id}.*.gii")
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_std_nomedialwall_roi_file(
        self, template_name: str, hemi: Literal["L", "R"], desc: Optional[str] = "nomedialwall"
    ) -> list[Path]:
        """
        Gets template nomedialwall ROI file.

        Args:
            template_name: Template name. The name shoule be defined in
                metadata file template.yaml.
            hemi: Brain hemisphere. Valid: L, R.
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.

        Returns:
            A list of template nomedialwall ROI file.
        """

        with open(self.get_template_info_file()[0], "r") as f:
            template_info = yaml.load(f, Loader=yaml.CLoader)
        if template_name not in template_info.keys():
            raise ValueError(f"Template {template_name} is not defined in the template.yaml file.")
        temp_dir = template_info[template_name]
        desc = f"{desc}*" if desc else ""
        file_list = sorted(
            self.base_dir.joinpath(temp_dir).glob(f"*hemi-{hemi}*{desc}_probseg.shape.gii")
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_std_subcortical_roi_file(
        self, template_name: str, space: str = "MNI152NLin6Asym"
    ) -> list[Path]:
        """
        Gets template subcortical ROI file.

        Args:
            template_name: Template name. The name shoule be defined in
                metadata file template.yaml.
            space: Image space.

        Returns:
            A list of template subcortical ROI file.
        """

        with open(self.get_template_info_file()[0], "r") as f:
            template_info = yaml.load(f, Loader=yaml.CLoader)
        if template_name not in template_info.keys():
            raise ValueError(f"Template {template_name} is not defined in the template.yaml file.")
        temp_dir = template_info[template_name]
        file_list = sorted(
            self.base_dir.joinpath(temp_dir).glob(f"*space-{space}*_desc-Subcortical_dseg.nii.gz")
        )
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_std_cifti_file(
        self, template_name: str, metric: str, desc: Optional[str] = None
    ) -> list[Path]:
        """
        Gets template CIFTI file.

        Args:
            template_name: Template name. The name shoule be defined in
                metadata file template.yaml.
            metric: Surface metric name. E.g., curv, dseg.
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.

        Returns:
            A list of template CIFTI file.
        """

        with open(self.get_template_info_file()[0], "r") as f:
            template_info = yaml.load(f, Loader=yaml.CLoader)
        if template_name not in template_info.keys():
            raise ValueError(f"Template {template_name} is not defined in the template.yaml file.")
        temp_dir = template_info[template_name]
        desc = f"{desc}*" if desc else ""
        file_list = sorted(self.base_dir.joinpath(temp_dir).glob(f"*{desc}_{metric}.*.nii"))
        _ = check_file_list(file_list, n=1)
        return file_list

    def get_atlas_file(
        self, atlas_id: str, space: str = "fsLR", sub_id: Optional[Union[str, int]] = None
    ) -> list[Path]:
        """Gets standard atlas file.

        Args:
            atlas_id: Atlas name.
            space: Atlas space.
            sub_id: Subject ID. Optional. Only required if the filename
                of the atlas contains subject-specific part.
        Returns:
            A list of atlas files corresponding to left and right brain
            hemisphere. Either file could be None. Both hemisphere could
            be the same file.

        Raises:
            KeyError: Given atlas_id or space is not found in atlas.yaml
                file.
        """

        if sub_id:
            sub_id = conform_sub_id(sub_id, with_prefix=True)
        with open(self.get_atlas_info_file()[0], "r") as f:
            atlas_info = yaml.load(f, Loader=yaml.CLoader)
        if space not in atlas_info.keys():
            raise KeyError(
                f"Atlas space {space} is not found in atlas.yaml file. "
                f"Possible value: {', '.join(atlas_info.keys())}."
            )
        if atlas_id not in atlas_info[space].keys():
            raise KeyError(
                f"Atlas {atlas_id} ({space} space) is not found in atlas.yaml file. "
                f"Possible value: {', '.join(atlas_info[space].keys())}."
            )

        atlas_file = []
        for hemi in ["L", "R"]:
            if hemi in atlas_info[space][atlas_id].keys():
                # allow subject specific atlas file
                fname = atlas_info[space][atlas_id][hemi].format(sub_id=sub_id)
                f = Path(self.base_dir, fname)
                _ = check_file_list([f], n=1)
                atlas_file.append(f)
            else:
                atlas_file.append(None)
        return atlas_file

    ##################
    # Utility function
    ##################

    def get_run_id(
        self,
        sub_id: Union[str, int],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        exclude: bool = False,
    ) -> list[str]:
        """Gets run list of a task.

        This function reads the default run list of a task from data
        validation metadata and excludes bad runs based on the metadata
        if requested.
        If there is no run_id available (task with only a single run),
        it will return [""].

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A list of run_id of a task.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        task_id = conform_task_id(task_id, with_prefix=True)
        default_list = self.task_info[task_id]["run_list"]
        exclude_info = self.task_info[task_id]["exclude"]
        # Check if the subject has excluded run
        bad_list = exclude_info[sub_id] if sub_id in exclude_info else []
        # Single run task
        if len(default_list) == 0:
            if (sub_id in exclude_info) and exclude:
                run_list = []
            else:
                run_list = [""]
        # Multiple runs without selection
        elif run_id is None:
            if exclude:
                run_list = [i for i in default_list if i not in bad_list]
            else:
                run_list = default_list
        # Multiple runs with selection
        else:
            run_list = run_id if isinstance(run_id, list) else [run_id]
            run_list = [conform_run_id(i, with_prefix=True) for i in run_list]
            if exclude:
                run_list = [i for i in run_list if i not in bad_list]
        return run_list


class Project(Layout):
    """Generic project class.

    This class provides a series of functions to find, read and
    manipulate project files.

    Attributes:
        bids_subject_list: All SubjectID in the BIDS dataset.
        subject_list: SubjectID after data exclusion.
        task_list: All TaskID in the BIDS dataset.
        task_info: A dict contains information of the task. Usually it
            contains information of data exclusion.
        func_regex: A regular expression to match a task related file.
    """

    def __init__(self, project_root_dir):
        super().__init__(project_root_dir)

    ###############
    # Tabular files
    ###############

    def read_beh(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        suffix: str = "events",
        modifier: Optional[str] = None,
        exclude: bool = False,
    ) -> pd.DataFrame:
        """Reads behavior file to dataframe.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            suffix: Filename suffix after modifier and before extexsion.
            modifier: Any possible filename modifier after session and
                before suffix part.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A dataframe of behavior data.
        """

        file_list = self.get_beh_file(
            sub_id,
            run_id=run_id,
            task_id=task_id,
            ses_id=ses_id,
            suffix=suffix,
            modifier=modifier,
            exclude=exclude,
        )
        df = [pd.read_csv(f, sep="\t") for f in file_list]
        df = pd.concat(df).reset_index(drop=True)

        return df

    def read_confound(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        confound_list: Optional[list[str]] = [
            "trans_x",
            "trans_y",
            "trans_z",
            "rot_x",
            "rot_y",
            "rot_z",
            "framewise_displacement",
            "std_dvars",
            "dvars",
            "rmsd",
            "global_signal",
            "csf",
            "white_matter",
            "csf_wm",
        ],
        exclude: bool = False,
    ) -> pd.DataFrame:
        """Reads confound regressor timeseries to dataframe.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            confound_list: A list of confound regressor names. If it's
                None, reading all columns except PCA component related
                ones.
            exclude: If true, exclude runs based on data validation
                metadata.

        Returns:
            A dataframe of confound regressors generated by fMRIPrep.
        """

        sub_id = conform_sub_id(sub_id, with_prefix=True)
        file_list = self.get_confound_file(
            sub_id, task_id, run_id=run_id, ses_id=ses_id, exclude=exclude
        )
        df = []
        for f in file_list:
            df_run = pd.read_csv(f, sep="\t")
            # add useful information
            df_run["sub_id"] = sub_id
            df_run["task_id"] = task_id
            df_run["run_id"] = re.search(self.func_regex, f.name).group("run_id")
            # filter out a/t comp columns
            col_list = df_run.columns.tolist()
            col_list = [i for i in df_run.columns.tolist()[:-3] if "comp_cor" not in i]
            df_run = df_run.loc[:, ["sub_id", "task_id", "run_id"] + col_list]
            df.append(df_run)
        df = pd.concat(df).reset_index(drop=True)
        # Select column if requested
        if confound_list:
            df = df.loc[:, ["sub_id", "task_id", "run_id"] + confound_list]
        return df

    ###############################
    # Preprocessed functional files
    ###############################

    def read_preproc_func_cifti(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "bold",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
        volume_as_img: bool = False,
        standardize: Optional[Literal["zscore"]] = None,
    ) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
        """Reads preprocessed func data from CIFTI file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., bold, mean.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.
            volume_as_img: If true, the volume part in the CIFTI image
                is extracted as a nib.nifti1.Nifti1Image object. If
                false, it's extracted as a numpy array.
            standardize: Standardize each vertex/voxel along the time
                dimension. Valid: zscore.

        Returns:
            A dict contains splitted CIFTI data. The keys are SurfaceL,
            SurfaceR, and Volume. If in_file is a list of filenames, the
            data will be concatenate along the time dimension (row).
        """

        file_list = self.get_preproc_func_cifti_file(
            sub_id,
            task_id,
            run_id=run_id,
            ses_id=ses_id,
            metric=metric,
            space=space,
            den=den,
            desc=desc,
            exclude=exclude,
        )
        return read_dtseries(
            file_list, volume_as_img=volume_as_img, standardize=standardize, dtype=np.float32
        )

    def read_preproc_func_cifti_roi(
        self,
        sub_id: Union[int, str],
        task_id: str,
        roi_mask: Union[
            dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
            list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
        ],
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "bold",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
        standardize: Optional[Literal["zscore"]] = None,
        single_data_array: bool = True,
    ) -> Union[
        np.ndarray,
        list[np.ndarray],
        dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
        list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
    ]:
        """Reads preprocessed func data within ROI from CIFTI file.

        This function could read multiple ROI data at once. It's faster
        than a explicit for loop, since this method only reads the whole
        data once. In that case, the ROI data will be in a list instead
        of a single numpy array of dict.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            roi_mask: A (list of) ROI mask dict. It is usually generated
                by the 'make_roi_from_spec' function.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., bold, mean.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.
            standardize: Standardize each vertex/voxel along the time
                dimension. Valid: zscore.
            single_data_array: If true, concatenate all parts into a
                single numpy array along columns. Order: SurfaceL,
                SurfaceR, Volume.

        Returns:
            Depending on the inputs, the returned ROI data could be in
            several format.
            If the 'single_data_array' option is True (default), the ROI
            data will be contained in a numpy array. If it's False, the
            ROI data will be in a dict like the roi_mask.
            If the 'roi_mask' is a list of ROI mask dict, the data of
            each ROI will be in a list, and the order is the same as the
            'roi_mask'.
            Data of multiple runs will always be concatenated along the
            time (row) dimension.
        """

        file_list = self.get_preproc_func_cifti_file(
            sub_id,
            task_id,
            run_id=run_id,
            ses_id=ses_id,
            metric=metric,
            space=space,
            den=den,
            desc=desc,
            exclude=exclude,
        )
        return read_dtseries_roi(
            file_list,
            roi_mask,
            standardize=standardize,
            single_data_array=single_data_array,
            dtype=np.float32,
        )

    ##############################
    # Singletrial estimation files
    ##############################

    def read_singletrial_response_cifti(
        self,
        sub_id: Union[int, str],
        task_id: str,
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "beta",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
        volume_as_img: bool = False,
        standardize: Optional[Literal["zscore"]] = None,
    ) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
        """Reads singletrial response estimation from CIFTI file.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., bold, mean.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.
            volume_as_img: If true, the volume part in the CIFTI image
                is extracted as a nib.nifti1.Nifti1Image object. If
                false, it's extracted as a numpy array.
            standardize: Standardize each vertex/voxel along the trial
                dimension. Valid: zscore.

        Returns:
            A dict contains splitted CIFTI data. The keys are SurfaceL,
            SurfaceR, and Volume. If in_file is a list of filenames, the
            data will be concatenate along the trial dimension (row).
        """

        file_list = self.get_singletrial_response_cifti_file(
            sub_id,
            task_id,
            run_id=run_id,
            ses_id=ses_id,
            metric=metric,
            space=space,
            den=den,
            desc=desc,
            exclude=exclude,
        )
        return read_dscalar(
            file_list, volume_as_img=volume_as_img, standardize=standardize, dtype=np.float32
        )

    def read_singletrial_response_cifti_roi(
        self,
        sub_id: Union[int, str],
        task_id: str,
        roi_mask: Union[
            dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
            list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
        ],
        run_id: Optional[Union[str, int, list[str], list[int]]] = None,
        ses_id: Optional[str] = None,
        metric: str = "beta",
        space: str = "fsLR",
        den: str = "32k",
        desc: Optional[str] = "sm0pt0",
        exclude: bool = False,
        standardize: Optional[Literal["zscore"]] = None,
        single_data_array: bool = True,
    ) -> Union[
        np.ndarray,
        list[np.ndarray],
        dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
        list[dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]],
    ]:
        """Reads singletrial response data within ROI from CIFTI file.

        This function could read multiple ROI data at once. It's faster
        than a explicit for loop, since this method only reads the whole
        data once. In that case, the ROI data will be in a list instead
        of a single numpy array of dict.

        Args:
            sub_id: SubjectID.
            task_id: TaskID.
            roi_mask: A (list of) ROI mask dict. It is usually generated
                by the 'make_roi_from_spec' function.
            run_id: RunID. Optional.
            ses_id: SessionID. Optional.
            metric: Functional metric name. E.g., bold, mean.
            space: Surface space name. E.g., fsLR
            mesh_den: Surface mesh density. E.g., 32k
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
            exclude: If true, exclude runs based on data validation
                metadata.
            standardize: Standardize each vertex/voxel along the trial
                dimension. Valid: zscore.
            single_data_array: If true, concatenate all parts into a
                single numpy array along columns. Order: SurfaceL,
                SurfaceR, Volume.

        Returns:
            Depending on the inputs, the returned ROI data could be in
            several format.
            If the 'single_data_array' option is True (default), the ROI
            data will be contained in a numpy array. If it's False, the
            ROI data will be in a dict like the roi_mask.
            If the 'roi_mask' is a list of ROI mask dict, the data of
            each ROI will be in a list, and the order is the same as the
            'roi_mask'.
            Data of multiple runs will always be concatenated along the
            time (row) dimension.
        """

        file_list = self.get_singletrial_response_cifti_file(
            sub_id,
            task_id,
            run_id=run_id,
            ses_id=ses_id,
            metric=metric,
            space=space,
            den=den,
            desc=desc,
            exclude=exclude,
        )
        return read_dscalar_roi(
            file_list,
            roi_mask,
            standardize=standardize,
            single_data_array=single_data_array,
            dtype=np.float32,
        )

    #############
    # ROI related
    #############

    def read_atlas_info(
        self, atlas_file: Optional[PathLike] = None
    ) -> dict[str, dict[str, dict[str, str]]]:
        """Reads atlas information.

        Args:
            atlas_info_file: An atlas information yaml file.
            If None, read the default file inside metadata directory.

        Returns:
            A dict contains atlas information. Each key represents a
            space where the atlas is in. The value is a dict mapping
            atlas name to its file information.
            For each atlas, the file information is stored in a dict
            which has two items (key: L, R) indicate files of left and
            right brain hemisphere (could be same file if the atlas file
            is bilateral).
        """

        if atlas_file is None:
            # Use default file if not provided
            atlas_file = self.get_atlas_info_file()[0]
        with open(atlas_file, "r") as f:
            atlas_info = yaml.load(f, Loader=yaml.CLoader)
        return atlas_info

    def read_roi_spec(self, roi_spec_file: Optional[PathLike] = None) -> dict[str, dict[str, Any]]:
        """Reads ROI specification.

        Args:
            roi_spec_file: A ROI specification yaml file.
            If None, read the default file insidemetadata directory.

        Returns
        -------
        dict
            A dict contains ROI specification. Each keys represents a
            ROI. The value is the specification which is also a dict.
            Common items in the dict are AtlasID, ROIType, IndexL,
            IndexR, LabelL and LabelR. There could be additional items
            present in the dict.
        """

        if roi_spec_file is None:
            # Use default file if not provided
            roi_spec_file = self.get_roi_definition_file()[0]
        with open(roi_spec_file, "r") as f:
            roi_spec = yaml.load(f, Loader=yaml.CLoader)
        return roi_spec

    def read_roi_list(self, roi_list_file: Optional[PathLike] = None) -> list[str]:
        """Reads ROI list.

        Args:
            roi_list_file: A ROI list yaml file.
            If None, read the default file inside metadata directory.

        Returns
            A list of ROI names.
        """

        if roi_list_file is None:
            # Use default file if not provided
            roi_list_file = self.get_roi_list_file()[0]
        with open(roi_list_file, "r") as f:
            roi_list = yaml.load(f, Loader=yaml.CLoader)
        return roi_list

    def make_roi_from_spec(
        self,
        roi_id: str,
        roi_spec: dict[str, dict[str, Any]],
        sub_id: Optional[Union[str, int]] = None,
        atlas_file: Optional[list[PathLike]] = None,
    ) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
        """Makes ROI mask based on given ROI specification.

        Args:
            roi_id: ROI name.
            roi_spec: ROI specification. Usually it generates by the
                'read_roi_spec' function.
            sub_id: Subject ID. Optional. Only required if the filename
                of the atlas contains subject-specific part.
            atlas_file: Atlas files used for creating ROI. Optional. By
                default, atlas files are retrieved by the
                'get_atlas_file' function. It should be a list of
                filenames corresponding to left and right brain
                hemisphere. Either file could be None. Both hemispheres
                could be the same file.

        Returns:
            A dict with 3 items. The keys are SurfaceL, SurfaceR and
            Volume, corresponding to the left, right brain hemisphere
            and the volume part. Usually a ROI could be either in
            surface or volume format, but not both.
            Surface mask is represented in a numpy array. Volume mask is
            represented in a nib.nifti1.Nifti1Image image.
        """

        roi_name, _ = parse_roi_id(roi_id)
        atlas_id = roi_spec[roi_name]["AtlasID"]
        space = roi_spec[roi_name]["AtlasSpace"]
        if atlas_file is None:
            atlas_file = self.get_atlas_file(atlas_id, space=space, sub_id=sub_id)
        roi_mask = make_roi_from_spec(roi_id, roi_spec, atlas_file)
        return roi_mask

    def unmask(
        self,
        data: np.ndarray,
        roi_mask: dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]],
    ) -> dict[str, Union[np.ndarray, nib.nifti1.Nifti1Image]]:
        """Reshapes ROI data back into its original shape.

        Args:
            data: Any data (a numpy array) generates by custom ROI data
                reading function in the package
                (e.g., read_preproc_func_cifti_roi).
            roi_mask: The ROI mask dict used to read the data.

        Returns:
            A dict with 3 items contains the unmasked data. The keys are
            SurfaceL, SurfaceR and Volume, corresponding to the left, right
            brain hemisphere and the volume part.
            Surface mask is represented in a numpy array. Volume mask is
            represented in a nib.nifti1.Nifti1Image image.

        Raises:
            ValueError: data is not a 1-d or 2d numpy array.
            ValueError: roi_mask is invalid.
        """

        return unmask(data, roi_mask)

    ##########################
    # Quick plotting functions
    ##########################

    def view_img(
        self,
        stat_map_img: nib.nifti1.Nifti1Image,
        bg_img: Union[
            Literal["T1w", "MNI152NLin2009cAsym", "MNI152NLin6Asym", "MNI152"],
            nib.nifti1.Nifti1Image,
        ] = "MNI152",
        sub_id: Optional[str] = None,
        ses_id: Optional[str] = None,
        modifier: Optional[str] = None,
        **kwargs,
    ) -> StatMapView:
        """Views a volume image.

        Args:
            stat_map_img: A nib.nifti1.Nifti1Image image.
            bg_img: Background image to plot on top of. It could be
                a nib.nifti1.Nifti1Image image object or a string. Valid
                string: T1w, MNI152NLin2009cAsym, MNI152NLin6Asym,
                MNI152. If it's MNI152, using nilearn's MNI152 template
                and ignoring argument sub_id, ses_id and modifier.
            sub_id: SubjectID. Optional.
            ses_id: SessionID. Optional.
            modifier: Any possible filename modifier after space and
                before suffix part. Optional see function
                'get_fmriprep_anat_file'.
            **kwargs: Additional keyword arguments pass to nilearn
                'view_img' function.

        Returns:
            A nilearn StatMapView object. It can be saved as an html
            page html_view.save_as_html('test.html'), or opened in a
            browser html_view.open_in_browser(). If the output is not
            requested and the current environment is a Jupyter notebook,
            the viewer will be inserted in the notebook.
        """

        import nilearn.plotting as nlp

        valid_list = ["T1w", "MNI152NLin2009cAsym", "MNI152NLin6Asym", "MNI152"]
        if isinstance(bg_img, str):
            if bg_img not in valid_list:
                raise ValueError(f"Unsupported bg_img. Valid: {', '.join(valid_list)}")
            if (bg_img != "MNI152") and (sub_id is None):
                raise ValueError(
                    "If a subject-specific bg_img is requested (e.g., T1w), "
                    "the sub_id must be specified."
                )

        # get subject specific bg_img
        if isinstance(bg_img, str) and (bg_img != "MNI152"):
            if bg_img == "T1w":
                print(
                    "Warning: The 'view_img' function seem to rely on a MNI template to display "
                    "bg_img.\nAs a result, the non-MNI image will be cropped. Usually this is not "
                    "a problem for a quick look of data."
                )
                bg_img = self.get_fmriprep_anat_file(sub_id, ses_id=ses_id, modifier=modifier)[0]
            else:
                bg_img = self.get_fmriprep_anat_file(
                    sub_id, ses_id=ses_id, space=bg_img, modifier=modifier
                )[0]
            bg_img = nib.load(bg_img)

        with warnings.catch_warnings():  # Ignore nilearn's UserWarning
            warnings.simplefilter("ignore")
            g = nlp.view_img(stat_map_img, bg_img=bg_img, dim=0, **kwargs)
        return g

    def view_surf_data(
        self,
        hemi: str,
        surf_map: Union[np.ndarray, PathLike],
        surf_mesh: Optional[
            Union[
                Literal["pial", "wm", "midthickness", "inflated", "veryinflated"],
                PathLike,
                list[np.ndarray],
                tuple[np.ndarray, np.ndarray],
            ]
        ] = "midthickness",
        template_name: str = "fsLR_32k",
        bg_map: Optional[Union[Literal["sulc"], np.ndarray, PathLike]] = "sulc",
        desc: Optional[str] = None,
        **kwargs,
    ) -> SurfaceView:
        """Views surface data.

        Args:
            hemi: Brain hemisphere. Valid: L, R.
            surf_map: A map need to be displayed on surface. The size of
                the surf_map should match the number of vertices in the
                surf_mesh.
            surf_mesh: Surface mesh used for displaying data. It could
                be a surface name in the template. It could also be a
                surface mesh filename or a list of two numpy array which
                represents a surface mesh.
            template_name: Template name of the surface mesh.
            bg_map: Background image to be plotted on the mesh
                underneath the surf_data in greyscale, most likely a
                sulcal depth map for realistic shading. It could
                be a surface metric in the template. It could also be a
                surface mesh filename or a numpy array with same size as
                the number of vertices in the surf_mesh.
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
                This argument will be pass to 'get_std_surf_file'
                function.
            **kwargs: Additional keyword arguments pass to nilearn
                'view_surf' function.

        Returns:
            A nilean SurfaceView object. It can be saved as an html
            page html_view.save_as_html('test.html'), or opened in a
            browser html_view.open_in_browser(). If the output is not
            requested and the current environment is a Jupyter notebook,
            the viewer will be inserted in the notebook.
        """

        import nilearn.plotting as nlp

        # surf_map
        surf_map = str(surf_map) if not isinstance(surf_map, np.ndarray) else surf_map
        # surface mesh
        if isinstance(surf_mesh, tuple):
            surf_mesh = list(surf_mesh)
        surf_mesh = str(surf_mesh) if not isinstance(surf_mesh, list) else surf_mesh
        valid_list = ["pial", "wm", "midthickness", "inflated", "veryinflated"]
        if surf_mesh in valid_list:
            surf_mesh = str(self.get_std_surf_file(template_name, hemi, surf_mesh, desc=desc)[0])
        # bg_map
        if bg_map == "sulc":
            bg_map = self.get_std_surf_file(template_name, hemi, bg_map, desc=desc)[0]
        bg_map = str(bg_map) if not isinstance(bg_map, np.ndarray) else bg_map

        with warnings.catch_warnings():  # Ignore nilearn's UserWarning
            warnings.simplefilter("ignore")
            g = nlp.view_surf(surf_mesh, surf_map=surf_map, bg_map=bg_map, **kwargs)
        return g

    def view_roi_img(
        self,
        roi_img: nib.nifti1.Nifti1Image,
        bg_img: Union[
            Literal["T1w", "MNI152NLin2009cAsym", "MNI152NLin6Asym", "MNI152"],
            nib.nifti1.Nifti1Image,
        ] = "MNI152",
        sub_id: Optional[str] = None,
        ses_id: Optional[str] = None,
        modifier: Optional[str] = None,
        threshold: float = 0.1,
        cmap: str = "Set1",
        symmetric_cmap: bool = False,
        colorbar: bool = False,
        vmax: float = 10,
        **kwargs,
    ) -> StatMapView:
        """Views a volume ROI.

        Args:
            roi_img: A nib.nifti1.Nifti1Image image.
            bg_img: Background image to plot on top of. It could be
                a nib.nifti1.Nifti1Image image object or a string. Valid
                string: T1w, MNI152NLin2009cAsym, MNI152NLin6Asym,
                MNI152. If it's MNI152, using nilearn's MNI152 template
                and ignoring argument sub_id, ses_id and modifier.
            sub_id: SubjectID. Optional.
            ses_id: SessionID. Optional.
            modifier: Any possible filename modifier after space and
                before suffix part. Optional see function
                'get_fmriprep_anat_file'.
            threshold: If None is given, the image is not thresholded.
                If a string of the form "90%" is given, use the 90-th
                percentile of the absolute value in the image. If a
                number is given, it is used to threshold the image:
                values below the threshold (in absolute value) are
                plotted as transparent. If auto is given, the threshold
                is determined automatically.
            cmap: The colormap for specified image.
            symmetric_cmap: True: make colormap symmetric (ranging from
                -vmax to vmax). False: the colormap will go from the
                minimum of the volume to vmax. Set it to False if you
                are plotting a positive volume, e.g. an atlas or an
                anatomical image.
            colorbar: If True, display a colorbar on top of the plots.
            vmax: Max value for mapping colors. If vmax is None and
                symmetric_cmap is True, vmax is the max absolute value
                of the volume. If vmax is None and symmetric_cmap is
                False, vmax is the max value of the volume.
                **kwargs: Additional keyword arguments pass to nilearn
                    'view_img' function.
            **kwargs: Additional keyword arguments pass to nilearn
                'view_img' function.

        Returns:
            A nilearn StatMapView object. It can be saved as an html
            page html_view.save_as_html('test.html'), or opened in a
            browser html_view.open_in_browser(). If the output is not
            requested and the current environment is a Jupyter notebook,
            the viewer will be inserted in the notebook.
        """

        return self.view_img(
            roi_img,
            bg_img=bg_img,
            sub_id=sub_id,
            ses_id=ses_id,
            modifier=modifier,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            vmax=vmax,
            resampling_interpolation="nearest",
            **kwargs,
        )

    def view_roi_surf(
        self,
        hemi: str,
        surf_map: Union[np.ndarray, PathLike],
        surf_mesh: Optional[
            Union[
                Literal["pial", "wm", "midthickness", "inflated", "veryinflated"],
                PathLike,
                list[np.ndarray],
                tuple[np.ndarray, np.ndarray],
            ]
        ] = "midthickness",
        template_name: str = "fsLR_32k",
        bg_map: Optional[Union[Literal["sulc"], np.ndarray, PathLike]] = "sulc",
        desc: Optional[str] = None,
        threshold: float = 0.1,
        cmap: str = "Set1",
        symmetric_cmap: bool = False,
        colorbar: bool = False,
        vmax: float = 10,
        **kwargs,
    ) -> SurfaceView:
        """Views surface data.

        Args:
            hemi: Brain hemisphere. Valid: L, R.
            surf_map: A map need to be displayed on surface. The size of
                the surf_map should match the number of vertices in the
                surf_mesh.
            surf_mesh: Surface mesh used for displaying data. It could
                be a surface name in the template. It could also be a
                surface mesh filename or a list of two numpy array which
                represents a surface mesh.
            template_name: Template name of the surface mesh.
            bg_map: Background image to be plotted on the mesh
                underneath the surf_data in greyscale, most likely a
                sulcal depth map for realistic shading. It could
                be a surface metric in the template. It could also be a
                surface mesh filename or a numpy array with same size as
                the number of vertices in the surf_mesh.
            desc: The desc part in the filename. It could be part of the
                full desc string, as long as it only matches one file.
                This argument will be pass to 'get_std_surf_file'
                function.
            threshold: If None is given, the image is not thresholded.
                If a string of the form "90%" is given, use the 90-th
                percentile of the absolute value in the image. If a
                number is given, it is used to threshold the image:
                values below the threshold (in absolute value) are
                plotted as transparent. If auto is given, the threshold
                is determined automatically.
            cmap: The colormap for specified image.
            symmetric_cmap: True: make colormap symmetric (ranging from
                -vmax to vmax). False: the colormap will go from the
                minimum of the volume to vmax. Set it to False if you
                are plotting a positive volume, e.g. an atlas or an
                anatomical image.
            colorbar: If True, display a colorbar on top of the plots.
            vmax: Max value for mapping colors. If vmax is None and
                symmetric_cmap is True, vmax is the max absolute value
                of the volume. If vmax is None and symmetric_cmap is
                False, vmax is the max value of the volume.
                **kwargs: Additional keyword arguments pass to nilearn
                    'view_img' function.
            **kwargs: Additional keyword arguments pass to nilearn
                'view_surf' function.

        Returns:
            A nilean SurfaceView object. It can be saved as an html
            page html_view.save_as_html('test.html'), or opened in a
            browser html_view.open_in_browser(). If the output is not
            requested and the current environment is a Jupyter notebook,
            the viewer will be inserted in the notebook.
        """

        return self.view_surf_data(
            hemi,
            surf_map,
            surf_mesh=surf_mesh,
            template_name=template_name,
            bg_map=bg_map,
            desc=desc,
            threshold=threshold,
            cmap=cmap,
            symmetric_cmap=symmetric_cmap,
            colorbar=colorbar,
            vmax=vmax,
            **kwargs,
        )
