#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Workflows to preprocess anatomical and functional data."""

# Author: Zhifang Ye
# Email: zhifang.ye.fghm@gmail.com
# Notes:
# - These preprocess workflows are intended to use after fMRIPrep
# - It generates HCP-like output files as well spec file for Workbench
# - Functional data is resampled to fsLR space with 32k mesh density and
#   MNI152NLin6Asym space with 2mm resolution for subcortical regions.
#   This is the canonical functional space in HCP 3T data.


from __future__ import annotations
from typing import Optional, Union, Literal

from pathlib import Path
import shutil
import re
import tempfile

from .template import copy_template_file
from .freesurfer import (
    convert_freesurfer_geometry_surface,
    convert_freesurfer_metric,
    convert_freesurfer_annot,
    convert_freesurfer_volume,
)
from .surface import (
    make_midthickness_surface,
    make_inflated_surface,
    make_nomedialwall_roi,
    refine_nomedialwall_roi,
    mask_metric_nomedialwall,
    smooth_metric,
)
from .registration import (
    warp_native_sphere_to_fsLR,
    calc_native_to_fsLR_registration_MSMSulc,
    calc_registration_distortion,
)
from .volume import make_brainmask_from_atlas, warp_atlas_to_reference, make_cortical_ribbon
from .bold import (
    find_good_voxel,
    sample_volume_to_surface,
    extract_func_subcortical,
    convert_fwhm_to_str,
    make_func_map_name,
)
from ..image.gifti import resample_surface, resample_metric, resample_label
from ..image.cifti import make_dense_scalar, make_dense_label, make_dense_timeseries
from ..utils.validation import (
    conform_sub_id,
    parse_hemi,
    parse_space,
    parse_mesh_density,
    parse_registration_method,
    parse_smoothing_fwhm,
)
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


class NativeSurface:
    """Native surfaces processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        fs_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        ses_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
            sub_id: SubjectID.
            fs_dir: Subject's FreeSurfer output directory.
            template_dir: Directory contains required template files.
            out_dir: Directory to store output file.
            ses_id: SessionID. Used in the filename prefix. For example,
                sub-001_ses-01.
            run_id: RunID. Used in the filename prefix. For example,
                sub-001_run-1.
        """

        #############
        # Directories
        #############
        self.sub_id = conform_sub_id(sub_id, with_prefix=False)
        self.fs_dir = Path(fs_dir)
        self.template_dir = Path(template_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        # Output filename prefix
        self.anat_prefix = f"sub-{self.sub_id}"
        if ses_id:
            self.anat_prefix += f"_ses-{ses_id}"
        if run_id:
            self.anat_prefix += f"_run-{run_id}"
        # Store important result files
        self.native = {"L": {}, "R": {}}

    def run_native_space_pipeline(
        self,
        hemi: Literal["L", "R"],
        xfm_file: Optional[PathLike] = None,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        msm_config_file: Optional[PathLike] = None,
        inflate_extra_scale: Union[float, int] = 1.0,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Runs native surface pipeline.

        Args:
            hemi: Brain hemisphere.
            xfm_file: An ITK format affine transformation matrix file.
                If it is given, applying it to the native surface files.
                Optional.
            registration_method: Surface-based registration method. If
                FS, use FreeSurfer's fsaverage registration to warp
                native surfaces to fsLR space. If MSMSulc, calculate
                native to fsLR from scratch using MSM program.
            msm_config_file: MSMSulc configuration file. Only required
                when registration_method is MSMSulc.
            inflate_extra_scale: Extra iteration scaling value. This
                value is used in function calc_inflation_scale to
                calculate the final iteration scaling value.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        res = {}
        # Convert FreeSurfer's native surfaces
        out = self.convert_native_surface(hemi, xfm_file=xfm_file, debug=debug)
        res.update(out)
        # Convert FreeSurfer's metric data
        out = self.convert_native_metric(hemi)
        res.update(out)
        # Convert FreeSurfer's annotation data
        out = self.convert_native_annotation(hemi)
        res.update(out)
        # Make midthinkness and inflated surfaces
        out = self.make_aux_native_surface(hemi, inflate_extra_scale=inflate_extra_scale)
        res.update(out)
        # Make (no)medialwall ROI by thresholding the surface thickness metric
        out = self.make_nomedialwall_roi(hemi)
        res.update(out)
        # Calculate registraion between native and fsLR space
        out = self.calc_native_to_fsLR_registration(
            hemi,
            registration_method=registration_method,
            msm_config_file=msm_config_file,
            debug=debug,
        )
        res.update(out)
        # Calculate registration distortion
        out = self.calc_registration_distortion(hemi, registration_method=registration_method)
        res.update(out)
        if registration_method != "FS":
            out = self.calc_registration_distortion(hemi, registration_method="FS")
            res.update(out)
        # Refine (no)medialwall ROI using template ROI
        out = self.refine_nomedialwall_roi(hemi, registration_method=registration_method)
        res.update(out)
        # Apply (no)medialwall ROI mask to metric data (curv, thickness)
        out = self.mask_metric_nomedialwall(hemi)
        res.update(out)

        return res

    def check_template_data(self):
        """Checks common template files in template_dir."""

        print("\n###Check template file###\n", flush=True)
        copy_template_file(self.template_dir, check_output_only=True)

    def convert_native_surface(
        self, hemi: Literal["L", "R"], xfm_file: Optional[PathLike] = None, debug: bool = False
    ) -> dict[str, Path]:
        """Converts FreeSurfer's native surfaces to GIFTI format.

        Args:
            hemi: Brain hemisphere.
            xfm_file: An ITK format affine transformation matrix file.
                If it is given, applying it to the native surface files.
                Optional.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi, structure_format="gifti")

        print(
            f"\n###Convert surfaces from FreeSurfer's reconstruction (hemi-{hemi})###\n",
            flush=True,
        )
        res = {}
        # white and pial surfaces
        for surf_id in ["white", "pial"]:
            out_file = convert_freesurfer_geometry_surface(
                self.sub_id,
                hemi,
                surf_id,
                self.fs_dir,
                self.out_dir,
                xfm_file=xfm_file,
                debug=debug,
            )
            # rename output file
            out_file = out_file.rename(
                Path(self.out_dir, out_file.name.replace(f"sub-{self.sub_id}", self.anat_prefix))
            )
            res[f"hemi-{hemi}_{surf_id}"] = out_file
            self.native[hemi][f"hemi-{hemi}_{surf_id}"] = out_file
        # sphere and sphere.reg
        for surf_id in ["sphere", "sphere.reg"]:
            out_file = convert_freesurfer_geometry_surface(
                self.sub_id,
                hemi,
                surf_id,
                self.fs_dir,
                self.out_dir,
                xfm_file=None,
                debug=debug,
            )
            # rename output file
            out_file = out_file.rename(
                Path(self.out_dir, out_file.name.replace(f"sub-{self.sub_id}", self.anat_prefix))
            )
            res[f"hemi-{hemi}_{surf_id}"] = out_file
            self.native[hemi][f"hemi-{hemi}_{surf_id}"] = out_file

        return res

    def convert_native_metric(self, hemi: Literal["L", "R"]) -> dict[str, Path]:
        """Converts FreeSurfer's surface metric to GIFTI format.

        Args:
            hemi: Brain hemisphere.

        Returns:
            A dict stores generated files.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        # Metric data: Sulc, curvature, cortical thickness
        print(f"\n###Convert FreeSurfer's surface metric (hemi-{hemi})###\n", flush=True)
        res = {}
        for metric_id in ["sulc", "curv", "thickness"]:
            out_file = convert_freesurfer_metric(
                self.sub_id, hemi, metric_id, self.fs_dir, self.out_dir
            )
            # rename output file
            out_file = out_file.rename(
                Path(self.out_dir, out_file.name.replace(f"sub-{self.sub_id}", self.anat_prefix))
            )
            res[f"hemi-{hemi}_{metric_id}"] = out_file
            self.native[hemi][f"hemi-{hemi}_{metric_id}"] = out_file

        return res

    def convert_native_annotation(self, hemi: Literal["L", "R"]) -> dict[str, Path]:
        """Converts FreeSurfer's annotation data to GIFTI format.

        Args:
            hemi: Brain hemisphere.

        Returns:
            A dict stores generated files.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        # Annotation: Aparc, Destrieux 2009, DKT
        print(f"\n###Convert FreeSurfer's annotation (hemi-{hemi})###\n", flush=True)
        res = {}
        for annot_id in ["aparc", "aparc.a2009s", "aparc.DKTatlas"]:
            out_file = convert_freesurfer_annot(
                self.sub_id, hemi, annot_id, self.fs_dir, self.out_dir
            )
            # rename output file
            out_file = out_file.rename(
                Path(self.out_dir, out_file.name.replace(f"sub-{self.sub_id}", self.anat_prefix))
            )
            res[f"hemi-{hemi}_{annot_id}"] = out_file
            self.native[hemi][f"hemi-{hemi}_{annot_id}"] = out_file

        return res

    def make_aux_native_surface(
        self, hemi: Literal["L", "R"], inflate_extra_scale: Union[float, int] = 1.0
    ) -> dict[str, Path]:
        """Makes midthinkness and inflated surfaces in native space.

        Args:
            hemi: Brain hemisphere.
            inflate_extra_scale: Extra iteration scaling value. This
                value is used in function calc_inflation_scale to
                calculate the final iteration scaling value.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        # Required files
        wm_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_wm.surf.gii"
        )
        pial_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_pial.surf.gii"
        )
        for f in [wm_file, pial_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface {f} not found. Run function 'prepare_native_surface' first."
                )

        # Output
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        inflated_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_inflated.surf.gii"
        )
        veryinflated_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_veryinflated.surf.gii"
        )

        res = {}
        # Midthickness surface
        print(
            f"\n###Create midthickness surface from white and pial surfaces (hemi-{hemi})###\n",
            flush=True,
        )
        out_file = make_midthickness_surface(hemi, wm_file, pial_file, midthickness_file)
        res[f"hemi-{hemi}_midthickness"] = out_file
        self.native[hemi][f"hemi-{hemi}_midthickness"] = out_file

        # Inflated and very inflated surfaces
        print(
            f"\n###Create inflated surfaces from midthickness surface (hemi-{hemi})###\n",
            flush=True,
        )
        out_file = make_inflated_surface(
            midthickness_file,
            inflated_file,
            veryinflated_file,
            inflate_extra_scale=inflate_extra_scale,
        )

        # Record result
        res[f"hemi-{hemi}_inflated"] = out_file[0]
        self.native[hemi][f"hemi-{hemi}_inflated"] = out_file[0]
        res[f"hemi-{hemi}_veryinflated"] = out_file[1]
        self.native[hemi][f"hemi-{hemi}_veryinflated"] = out_file[1]

        return res

    def make_nomedialwall_roi(self, hemi: Literal["L", "R"]) -> dict[str, Path]:
        """Makes (no)medialwall ROI by thresholding the surface thickness.

        Args:
            hemi: Brain hemisphere.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        # Required files
        thickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_thickness.shape.gii"
        )
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        if not thickness_file.is_file():
            raise FileNotFoundError(
                f"Metric {thickness_file} not found. Run function 'prepare_native_metric' first."
            )
        if not midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {midthickness_file} not found. "
                "Run function 'make_aux_native_surface' first."
            )

        # Output
        out_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        gifti_map_name = f"{self.anat_prefix}_hemi-{hemi}_desc-nomedialwall"

        # Make ROI
        res = {}
        print(f"\n###Create (no)medialwall ROI (hemi-{hemi})###\n", flush=True)
        out_file = make_nomedialwall_roi(
            thickness_file, midthickness_file, out_file, gifti_map_name=gifti_map_name
        )

        # Record result
        res[f"hemi-{hemi}_nomedialwall"] = out_file
        self.native[hemi][f"hemi-{hemi}_nomedialwall"] = out_file

        return res

    def calc_native_to_fsLR_registration(
        self,
        hemi: Literal["L", "R"],
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        msm_config_file: Optional[PathLike] = None,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Calculates registration between native and fsLR space.

        Args:
            hemi: Brain hemisphere.
            registration_method: Surface-based registration method. If
                FS, use FreeSurfer's fsaverage registration to warp
                native surfaces to fsLR space. If MSMSulc, calculate
                native to fsLR from scratch using MSM program.
            msm_config_file: MSMSulc configuration file. Only required
                when registration_method is MSMSulc.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        # native sphere mesh mesh in fsaverage space (generated by FreeSurfer)
        in_reg_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsaverage_den-fsnative_sphere.surf.gii"
        )
        # template sphere mesh in fsaverage space
        proj_sphere_file = self.template_dir.joinpath(
            f"fsaverage_hemi-{hemi}_space-fsaverage_den-164k_sphere.surf.gii"
        )
        # template fsaverage sphere mesh in fsLR space
        unproj_sphere_file = self.template_dir.joinpath(
            f"fsaverage_hemi-{hemi}_space-fsLR_den-164k_sphere.surf.gii"
        )
        if not in_reg_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {in_reg_sphere_file} is not found. "
                "Run function 'prepare_native_surface' first."
            )
        for f in [proj_sphere_file, unproj_sphere_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Sphere mesh {f} is not found. " "Run function 'copy_template_data' first."
                )
        # for MSMSulc registration
        if method == "MSMSulc":
            # native mesh in native space
            in_sphere_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_sphere.surf.gii"
            )
            # metric sulc in native space
            in_sulc_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_sulc.shape.gii"
            )
            # template sphere mesh in fsLR space
            ref_sphere_file = self.template_dir.joinpath(
                f"fsLR_hemi-{hemi}_space-fsLR_den-164k_sphere.surf.gii"
            )
            # template metric sulc in fsLR space
            ref_sulc_file = self.template_dir.joinpath(
                f"fsLR_hemi-{hemi}_space-fsLR_den-164k_sulc.shape.gii"
            )
            if not in_sphere_file.is_file():
                raise FileNotFoundError(
                    f"Sphere mesh {in_sphere_file} is not found."
                    "Run function 'prepare_native_surface' first."
                )
            if not in_sulc_file.is_file():
                raise FileNotFoundError(
                    f"Metric {in_sulc_file} is not found. "
                    "Run function 'prepare_native_metric' first."
                )
            if not ref_sphere_file.is_file():
                raise FileNotFoundError(
                    f"Sphere mesh {ref_sphere_file} is not found. "
                    "Run function 'copy_template_data' first."
                )
            if not ref_sulc_file.is_file():
                raise FileNotFoundError(
                    f"Metric {ref_sulc_file} is not found. "
                    "Run function 'copy_template_data' first."
                )

        # Output
        # native mesh in fsLR space
        out_fs_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-FS_sphere.surf.gii"
        )
        out_msm_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-MSMSulc_sphere.surf.gii"
        )

        # Calculate registration using FreeSurfer method
        print(
            f"\n###Warp native sphere to fsLR using FreeSurfer's registration (hemi-{hemi})###\n",
            flush=True,
        )
        res = {}
        out_fs_file = warp_native_sphere_to_fsLR(
            in_reg_sphere_file, proj_sphere_file, unproj_sphere_file, out_fs_file
        )
        res[f"hemi-{hemi}_registration_FS"] = out_fs_file
        self.native[hemi][f"hemi-{hemi}_registration_FS"] = out_fs_file

        # Calculate registration using MSMSulc method
        if method == "MSMSulc":
            print(
                f"\n###Run registration to fsLR space using MSMSulc method (hemi-{hemi})###\n",
                flush=True,
            )
            out_msm_file = calc_native_to_fsLR_registration_MSMSulc(
                hemi,
                in_sphere_file,
                out_fs_file,
                in_sulc_file,
                ref_sphere_file,
                ref_sulc_file,
                out_msm_file,
                msm_config_file,
                debug=debug,
            )
            res[f"hemi-{hemi}_registration_MSMSulc"] = out_msm_file
            self.native[hemi][f"hemi-{hemi}_registration_MSMSulc"] = out_msm_file

        return res

    def calc_registration_distortion(
        self, hemi: Literal["L", "R"], registration_method: Literal["FS", "MSMSulc"] = "MSMSulc"
    ) -> dict[str, Path]:
        """Calculates registration distortion.

        Args:
            hemi: Brain hemisphere.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        src_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_sphere.surf.gii"
        )
        warpped_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        if not src_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {src_sphere_file} is not found. "
                "Run function 'prepare_native_surface' first."
            )
        if not warpped_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {warpped_sphere_file} is not found. "
                "Run function 'calc_native_to_fsLR_registration' first."
            )

        # Calculate 4 types of distortion (Areal, Edge, StrainJ, StrainR)
        print(f"\n###Calculate registration distortion measurement (hemi-{hemi})###\n", flush=True)
        res = {}
        for metric_id in ["Areal", "Edge", "StrainJ", "StrainR"]:

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
                f"desc-{method}-{metric_id}_distortion.shape.gii"
            ).as_posix()
            gifti_map_name = f"{self.anat_prefix}_hemi-{hemi}_desc-{method}-{metric_id}"

            # Calculate registration distortion
            out_file = calc_registration_distortion(
                src_sphere_file,
                warpped_sphere_file,
                out_file,
                metric_id,
                gifti_map_name=gifti_map_name,
            )

            # Record result
            res[f"hemi-{hemi}_distortion-{metric_id}_{method}"] = out_file
            self.native[hemi][f"hemi-{hemi}_distortion-{metric_id}_{method}"] = out_file

        return res

    def refine_nomedialwall_roi(
        self, hemi: Literal["L", "R"], registration_method: Literal["FS", "MSMSulc"] = "MSMSulc"
    ) -> dict[str, Path]:
        """Refines (no)medialwall ROI using template ROI.

        Args:
            hemi: Brain hemisphere.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        roi_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        warpped_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-164k_desc-nomedialwall_probseg.shape.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-164k_sphere.surf.gii"
        )
        if not roi_file.is_file():
            raise FileNotFoundError(
                f"Nomedialwall ROI {roi_file} is not found. "
                "Run function 'make_nomedialwall_roi' first."
            )
        if not warpped_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {warpped_sphere_file} is not found. "
                "Run function 'calc_native_to_fsLR_registration' first."
            )
        if not template_roi_file.is_file():
            raise FileNotFoundError(
                f"Nomedialwall ROI {template_roi_file} is not found. "
                "Run function 'copy_template_data' first."
            )
        if not template_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {template_sphere_file} is not found. "
                "Run function 'copy_template_data' first."
            )

        # Output
        out_file = roi_file

        # Refine ROI
        print(f"\n###Refine nomedialwall ROI (hemi-{hemi})###\n", flush=True)
        res = {}
        out_file = refine_nomedialwall_roi(
            roi_file, warpped_sphere_file, template_roi_file, template_sphere_file, out_file
        )

        # Record result
        res[f"hemi-{hemi}_nomedialwall"] = out_file
        self.native[hemi][f"hemi-{hemi}_nomedialwall"] = out_file

        return res

    def mask_metric_nomedialwall(self, hemi: Literal["L", "R"]) -> dict[str, Path]:
        """Applies (no)medialwall mask to native surface metric file.

        Args:
            hemi: Brain hemisphere.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)

        print(f"\n###Apply (no)medialwall ROI mask to metrics (hemi-{hemi})###\n", flush=True)
        res = {}
        for metric_id in ["curv", "thickness"]:

            # Required files
            metric_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_{metric_id}.shape.gii"
            )
            roi_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
                "desc-nomedialwall_probseg.shape.gii"
            )
            midthickness_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
            )
            if not metric_file.is_file():
                raise FileNotFoundError(
                    f"Surface metric {metric_file} is not found. "
                    "Run function 'prepare_native_metric' first."
                )
            if not roi_file.is_file():
                raise FileNotFoundError(
                    f"Nomedialwall ROI {roi_file} is not found. "
                    "Run function 'make_nomedialwall_roi' and 'refine_nomedialwall_roi' first."
                )
            if not midthickness_file.is_file():
                raise FileNotFoundError(
                    f"Surface {midthickness_file} is not found. "
                    "Run function 'make_aux_native_surface' first."
                )

            # Output
            out_file = metric_file

            # Apply (no)medialwall ROI
            out_file = mask_metric_nomedialwall(metric_file, roi_file, midthickness_file, out_file)

            # Record result
            res[f"hemi-{hemi}_{metric_id}"] = out_file
            self.native[hemi][f"hemi-{hemi}_{metric_id}"] = out_file

        return res


class ResampleSurface(NativeSurface):
    """Native to fsLR space processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        fs_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        ses_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
            sub_id: SubjectID.
            fs_dir: Subject's FreeSurfer output directory.
            template_dir: Directory contains required template files.
            out_dir: Directory to store output file.
            ses_id: SessionID. Used in the filename prefix. For example,
                sub-001_ses-01.
            run_id: RunID. Used in the filename prefix. For example,
                sub-001_run-1.
        """

        super().__init__(sub_id, fs_dir, template_dir, out_dir, ses_id=ses_id, run_id=run_id)
        # Store important result files
        self.fsLR = {"L": {}, "R": {}}

    def run_resample_fsLR_pipeline(
        self,
        hemi: Literal["L", "R"],
        target_mesh_density: list[str] = ["164k", "32k"],
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        inflate_extra_scale: Union[float, int] = 1.0,
    ) -> dict[str, Path]:
        """Runs resample native to fsLR space pipeline.

        Args:
            hemi: Brain hemisphere.
            traget_mesh_density: A list of surface mesh density which
                the native space will be resampled to.
            registration_method: Surface-based registration method.
            inflate_extra_scale: Extra iteration scaling value. This
                value is used in function calc_inflation_scale to
                calculate the final iteration scaling value.

        Returns:
            A dict stores generated files.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        for mesh_den in target_mesh_density:
            _ = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])

        res = {}
        # Resample surface mesh to fsLR space (164k, 32k)
        for mesh_den in target_mesh_density:
            out = self.resample_native_surface_to_fsLR(
                hemi, mesh_den, registration_method=registration_method
            )
            res.update(out)
        # Make inflated and veryinflated surface in fsLR space (164k, 32k)
        for mesh_den in target_mesh_density:
            out = self.make_aux_fsLR_surface(
                hemi,
                mesh_den,
                registration_method=registration_method,
                inflate_extra_scale=inflate_extra_scale,
            )
            res.update(out)
        # Resample metric data to fsLR space (164k, 32k)
        for mesh_den in target_mesh_density:
            out = self.resample_native_metric_to_fsLR(
                hemi,
                mesh_den,
                registration_method=registration_method,
            )
            res.update(out)
        # Resample registration distortion to fsLR space (164k, 32k)
        for mesh_den in target_mesh_density:
            out = self.resample_native_distortion_to_fsLR(
                hemi,
                mesh_den,
                registration_method=registration_method,
            )
            res.update(out)
        # Resample atlas to fsLR space (164k, 32k)
        for mesh_den in target_mesh_density:
            out = self.resample_native_label_to_fsLR(
                hemi,
                mesh_den,
                registration_method=registration_method,
            )
            res.update(out)

        return res

    def resample_native_surface_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native surfaces to fsLR space

        Args:
            hemi: Brain hemisphere.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Common required files
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        if not template_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {template_sphere_file} is not found. "
                "Run function 'copy_template_data' first."
            )

        print(f"\n###Resample native surfaces to fsLR space (hemi-{hemi})###\n", flush=True)
        res = {}
        for surf_id in ["wm", "pial", "midthickness"]:

            # Required files
            surf_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_{surf_id}.surf.gii"
            )
            warpped_sphere_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_"
                f"desc-{method}_sphere.surf.gii"
            )
            if not surf_file.is_file():
                raise FileNotFoundError(
                    f"Surface {surf_file} is not found. "
                    "Run function 'prepare_native_surface' first."
                )
            if not warpped_sphere_file.is_file():
                raise FileNotFoundError(
                    f"Sphere mesh {warpped_sphere_file} is not found. "
                    "Run function 'calc_native_to_fsLR_registration' first."
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
                f"desc-{method}_{surf_id}.surf.gii"
            )

            # Resample surface
            print(f"Resampling surface to fsLR {mesh_den} space: {surf_file} ...", flush=True)
            out_file = resample_surface(
                surf_file, warpped_sphere_file, template_sphere_file, out_file
            )

            # Record result
            res[f"hemi-{hemi}_{surf_id}_{mesh_den}_{method}"] = out_file
            self.fsLR[hemi][f"hemi-{hemi}_{surf_id}_{mesh_den}_{method}"] = out_file

        # Also create midthickness surface using FS registration if the main method is MSMSulc.
        # This surface is required to resample registration distortion files using FS method.
        if method != "FS":

            # Required files
            surf_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
            )
            warpped_sphere_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-FS_sphere.surf.gii"
            )

            if not surf_file.is_file():
                raise FileNotFoundError(
                    f"Surface {surf_file} is not found. "
                    "Run function 'prepare_native_surface' first."
                )
            if not warpped_sphere_file.is_file():
                raise FileNotFoundError(
                    f"Sphere mesh {warpped_sphere_file} is not found. "
                    "Run function 'calc_native_to_fsLR_registration' first."
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
                f"desc-FS_midthickness.surf.gii"
            )

            # Resample surface
            print(f"Resampling surface to fsLR {mesh_den} space: {surf_file} ...", flush=True)
            out_file = resample_surface(
                surf_file, warpped_sphere_file, template_sphere_file, out_file
            )

            # Record result
            res[f"hemi-{hemi}_{surf_id}_{mesh_den}_FS"] = out_file
            self.fsLR[hemi][f"hemi-{hemi}_{surf_id}_{mesh_den}_FS"] = out_file

        return res

    def make_aux_fsLR_surface(
        self,
        hemi: Literal["L", "R"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        inflate_extra_scale: Union[float, int] = 1.0,
    ) -> dict[str, Path]:
        """Makes inflated surfaces in fsLR space.

        Args:
            hemi: Brain hemisphere.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.
            inflate_extra_scale: Extra iteration scaling value. This
                value is used in function calc_inflation_scale to
                calculate the final iteration scaling value.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        if not midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {midthickness_file} is not found. "
                "Run function 'resample_native_surface_to_fsLR' first."
            )

        # Output
        inflated_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_inflated.surf.gii"
        )
        veryinflated_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_veryinflated.surf.gii"
        )

        # Make inflated and very inflated surface
        print(f"\n###Create inflated surfaces in fsLR space (hemi-{hemi})###\n", flush=True)
        res = {}
        out_file = make_inflated_surface(
            midthickness_file,
            inflated_file,
            veryinflated_file,
            inflate_extra_scale=inflate_extra_scale,
        )

        # Record result
        res[f"hemi-{hemi}_inflated_{mesh_den}_{method}"] = out_file[0]
        self.fsLR[hemi][f"hemi-{hemi}_inflated_{mesh_den}_{method}"] = out_file[0]
        res[f"hemi-{hemi}_veryinflated_{mesh_den}_{method}"] = out_file[1]
        self.fsLR[hemi][f"hemi-{hemi}_veryinflated_{mesh_den}_{method}"] = out_file[1]

        return res

    def resample_native_metric_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native surface metric to fsLR space.

        Args:
            hemi: Brain hemisphere.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Common required files
        warpped_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        warpped_midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        roi_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        template_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        if not warpped_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {warpped_sphere_file} is not found. "
                "Run function 'calc_native_to_fsLR_registration' first."
            )
        if not template_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {template_sphere_file} is not found. "
                "Run function 'copy_template_data' first."
            )
        if not midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {midthickness_file} is not found. "
                "Run function 'make_aux_native_surface' first."
            )
        if not warpped_midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {warpped_midthickness_file} is not found. "
                "Run function 'resample_native_surface_to_fsLR' first."
            )
        if not roi_file.is_file():
            raise FileNotFoundError(
                f"Nomedialwall ROI {roi_file} is not found. "
                "Run function 'make_nomedialwall_roi' and 'refine_nomedialwall_roi' first."
            )
        if not template_roi_file.is_file():
            raise FileNotFoundError(
                f"Nomedialwall ROI {template_roi_file} is not found. "
                "Run function 'copy_template_data' first."
            )

        print(f"\n###Resample metric data to fsLR space (hemi-{hemi})###\n", flush=True)
        res = {}
        for metric_id in ["sulc", "curv", "thickness"]:

            # Required file
            metric_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_{metric_id}.shape.gii"
            )
            if not metric_file.is_file():
                raise FileNotFoundError(
                    f"Surface metric {metric_file} is not found. "
                    "Run function 'prepare_native_metric' first."
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
                f"desc-{method}_{metric_id}.shape.gii"
            )

            # Resample metric
            print(f"Resampling metric to fsLR {mesh_den} space: {metric_file} ...", flush=True)
            if metric_id in ["curv", "thickness"]:
                out_file = resample_metric(
                    metric_file,
                    warpped_sphere_file,
                    template_sphere_file,
                    out_file,
                    current_area_surf_file=midthickness_file,
                    target_area_surf_file=warpped_midthickness_file,
                    roi_file=roi_file,
                    resample_method="ADAP_BARY_AREA",
                )
                run_cmd(
                    "wb_command -disable-provenance -metric-mask "
                    f"{out_file} {template_roi_file} {out_file}"
                )
            if metric_id == "sulc":
                out_file = resample_metric(
                    metric_file,
                    warpped_sphere_file,
                    template_sphere_file,
                    out_file,
                    current_area_surf_file=midthickness_file,
                    target_area_surf_file=warpped_midthickness_file,
                    resample_method="ADAP_BARY_AREA",
                )

            # Record result
            res[f"hemi-{hemi}_{metric_id}_{mesh_den}_{method}"] = out_file
            self.fsLR[hemi][f"hemi-{hemi}_{metric_id}_{mesh_den}_{method}"] = out_file

        return res

    def resample_native_distortion_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native registration distortion to fsLR space.

        Args:
            hemi: Brain hemisphere.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        # resample FS method registration distortion, if the main method is not
        method = parse_registration_method(registration_method)
        method_list = [method, "FS"] if method != "FS" else [method]

        # Common required files
        warpped_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        warpped_midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        if not warpped_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {warpped_sphere_file} is not found. "
                "Run function 'calc_native_to_fsLR_registration' first."
            )
        if not template_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {template_sphere_file} is not found. "
                "Run function 'copy_template_data' first."
            )
        if not midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {midthickness_file} is not found. "
                "Run function 'make_aux_native_surface' first."
            )
        if not warpped_midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {warpped_midthickness_file} is not found. "
                "Run function 'resample_native_surface_to_fsLR' first."
            )

        print(
            f"\n###Resample registration distortion to fsLR space (hemi-{hemi})###\n", flush=True
        )
        res = {}
        for method in method_list:
            for metric_id in ["Areal", "Edge", "StrainJ", "StrainR"]:

                # Required file
                metric_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
                    f"desc-{method}-{metric_id}_distortion.shape.gii"
                )
                if not metric_file.is_file():
                    raise FileNotFoundError(
                        f"Distortion metric {metric_file} is not found. "
                        "Run function 'calc_registration_distortion' first."
                    )

                # Output
                out_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
                    f"desc-{method}-{metric_id}_distortion.shape.gii"
                )

                # Resample metric
                print(f"Resampling metric to fsLR {mesh_den} space: {metric_file} ...", flush=True)
                out_file = resample_metric(
                    metric_file,
                    warpped_sphere_file,
                    template_sphere_file,
                    out_file,
                    current_area_surf_file=midthickness_file,
                    target_area_surf_file=warpped_midthickness_file,
                    resample_method="ADAP_BARY_AREA",
                )

                # Record result
                res[f"hemi-{hemi}_{metric_id}_{mesh_den}_{method}"] = out_file
                self.fsLR[hemi][f"hemi-{hemi}_{metric_id}_{mesh_den}_{method}"] = out_file

        return res

    def resample_native_label_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native surface annotation to fsLR space.

        Args:
            hemi: Brain hemisphere.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Common required files
        warpped_sphere_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        warpped_midthickness_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        if not warpped_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {warpped_sphere_file} is not found. "
                "Run function 'calc_native_to_fsLR_registration' first."
            )
        if not template_sphere_file.is_file():
            raise FileNotFoundError(
                f"Sphere mesh {template_sphere_file} is not found. "
                "Run function 'copy_template_data' first."
            )
        if not midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {midthickness_file} is not found. "
                "Run function 'make_aux_native_surface' first."
            )
        if not warpped_midthickness_file.is_file():
            raise FileNotFoundError(
                f"Surface {warpped_midthickness_file} is not found. "
                "Run function 'resample_native_surface_to_fsLR' first."
            )

        print(f"\n###Resample atlas to fsLR space (hemi-{hemi})###\n", flush=True)
        res = {}
        for atlas_id in ["Aparc", "Destrieux", "DKT"]:

            # Required file
            atlas_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
                f"desc-{atlas_id}_dseg.label.gii"
            )
            if not atlas_file.is_file():
                raise FileNotFoundError(
                    f"Atlas {atlas_file} is not found. "
                    "Run function 'prepare_native_annotation' first."
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
                f"desc-{method}-{atlas_id}_dseg.label.gii"
            )

            # Resample label
            print(f"Resampling atlas to fsLR {mesh_den} space: {atlas_file} ...", flush=True)
            out_file = resample_label(
                atlas_file,
                warpped_sphere_file,
                template_sphere_file,
                out_file,
                current_area_surf_file=midthickness_file,
                target_area_surf_file=warpped_midthickness_file,
                resample_method="ADAP_BARY_AREA",
            )

            # Record result
            res[f"hemi-{hemi}_{atlas_id}_{mesh_den}_{method}"] = out_file
            self.fsLR[hemi][f"hemi-{hemi}_{atlas_id}_{mesh_den}_{method}"] = out_file

        return res


class FreeSurferVolume(ResampleSurface):
    """FreeSurfer volume processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        fs_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        ses_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
                sub_id: SubjectID.
                fs_dir: Subject's FreeSurfer output directory.
                template_dir: Directory contains required template files.
                out_dir: Directory to store output file.
                ses_id: SessionID. Used in the filename prefix. For example,
                    sub-001_ses-01.
                run_id: RunID. Used in the filename prefix. For example,
                    sub-001_run-1.
        """

        super().__init__(sub_id, fs_dir, template_dir, out_dir, ses_id=ses_id, run_id=run_id)
        # Store important result files
        self.volume = {}

    def run_volume_pipeline(
        self,
        xfm_file: PathLike,
        ref_file: PathLike,
        xfm_mni_file: PathLike,
        ref_mni_file: PathLike,
        lut_file: PathLike,
        lut_subcortical_file: PathLike,
    ) -> dict[str, Path]:
        """Runs FreeSurfer volume pipeline.

        Args:
            xfm_file: An ITK format affine transformation matrix file.
                Usually it is used to adjust the difference between the
                original and FreeSurfer conformed T1w images.
            ref_file: Reference volume file for xfm_file.
            xfm_mni_file: An ITK format nonlinear transformation matrix
                file. It tranforms image in T1w space to MNI152NLin6Asym
                space.
            ref_mni_file: Reference volume file for xfm_mni_file.
            lut_file: Lut file contains label information of
                FreeSurfer's parcellations. It is used to import label
                information to parcellation NIFTI image header.
            lut_subcortical_file: Lut file contains label information of
                FreeSurfer's subcortical segmentation. It is used to
                import label information to parcellation NIFTI image
                header.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The ref_mni_file is not in MNI152NLin6Asym_res-2
                space.
        """

        res = {}
        # Convert FreeSurfer's volume to NIFTI format
        out = self.convert_freesurfer_volume(
            xfm_file=xfm_file, ref_file=ref_file, lut_file=lut_file
        )
        res.update(out)
        # Make brainmask
        out = self.make_brainmask_from_wmparc()
        res.update(out)
        # Warp subcortrical ROI to MNI space
        # check MNI reference file
        if not "MNI152NLin6Asym_res-2" in Path(ref_mni_file).name:
            raise ValueError("MNI reference file should be in MNI152NLin6Asym_res-2 space.")
        out = self.warp_subcortical_roi(
            xfm_mni_file,
            ref_mni_file,
            lut_file=lut_subcortical_file,
            space="MNI152NLin6Asym_res-2",
        )
        res.update(out)

        return res

    def convert_freesurfer_volume(
        self,
        xfm_file: Optional[PathLike] = None,
        ref_file: Optional[PathLike] = None,
        lut_file: Optional[PathLike] = None,
    ) -> dict[str, Path]:
        """Converts FreeSurfer's volume to NIFTI format.

        Note: FreeSurfer's interal anatomical image is in a conformed
        space. This space might has small differences to the input T1w
        image. This could be mitigate by using the fsnative to T1w
        transformation matrix generated by fMRIPrep.

        Args:
            xfm_file: An ITK format affine transformation matrix file.
                If it is given, applying it to the volume file.
                Optional.
            ref_file: Reference volume file for xfm_file. Optional.
            lut_file: Lut file contains label information of
                FreeSurfer's parcellations. It is used to import label
                information to parcellation NIFTI image header.
                Optional.

        Returns:
            A dict stores generated files.
        """

        print(f"\n###Convert FreeSurfer's volume###\n", flush=True)
        volume_list = ["T1", "wmparc", "aparc.a2009s+aseg", "aparc+aseg", "aparc.DKTatlas+aseg"]
        res = {}
        for volume_id in volume_list:
            out_file = convert_freesurfer_volume(
                self.sub_id,
                volume_id,
                self.fs_dir,
                self.out_dir,
                xfm_file=xfm_file,
                ref_file=ref_file,
                lut_file=lut_file,
            )
            # rename output file
            out_file = out_file.rename(
                Path(self.out_dir, out_file.name.replace(f"sub-{self.sub_id}", self.anat_prefix))
            )
            res[volume_id], self.volume[volume_id] = out_file, out_file

        return res

    def make_brainmask_from_wmparc(self) -> dict[str, Path]:
        """Makes brainmask from FreeSurfer's wmparc file."""

        # Required file
        atlas_file = self.out_dir.joinpath(f"{self.anat_prefix}_space-T1w_desc-WMParc_dseg.nii.gz")
        if not atlas_file.is_file():
            raise FileNotFoundError(
                f"File {atlas_file} is not found. " "Run 'convert_freesurfer_volume' first."
            )
        # Output
        out_file = self.out_dir.joinpath(f"{self.anat_prefix}_space-T1w_desc-brain-FS_mask.nii.gz")
        # Make brainmask
        print(f"\n###Make brainmask from WMParc###\n", flush=True)
        res = {}
        out_file = make_brainmask_from_atlas(atlas_file, out_file)
        res["brainmask"], self.volume["brainmask"] = out_file, out_file

        return res

    def warp_subcortical_roi(
        self,
        xfm_file: PathLike,
        ref_file: PathLike,
        lut_file: Optional[PathLike] = None,
        space: str = "MNI152NLin6Asym_res-2",
    ) -> dict[str, Path]:
        """Warps FreeSurfer's segmentation to target space.

        Note: FreeSurfer's interal anatomical image is in a conformed
        space. This space might has small differences to the input T1w
        image. This could be mitigate by using the fsnative to T1w
        transformation matrix generated by fMRIPrep.

        Args:
            xfm_file: An ITK format affine transformation matrix file.
                If it is given, applying it to the volume file.
            ref_file: Reference volume file for xfm_file.
            lut_file: Lut file contains label information of
                FreeSurfer's parcellations. It is used to import label
                information to parcellation NIFTI image header.
                Optional.
            space: Spatial space of the output file. It should match the
                given transformation xfm_file.

        Returns:
            A dict stores generated files.

        Raises:
            FileNotFoundError: Required file is not found.
        """

        # Required file
        atlas_file = self.out_dir.joinpath(f"{self.anat_prefix}_space-T1w_desc-WMParc_dseg.nii.gz")
        if not atlas_file.is_file():
            raise FileNotFoundError(
                f"File {atlas_file} is not found. " "Run 'convert_freesurfer_volume' first."
            )
        # Output
        out_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_space-{space}_desc-Subcortical_dseg.nii.gz"
        )
        # Warp file
        print(f"\n###Warp subcortical ROI to {space} space###\n", flush=True)
        res = {}
        out_file = warp_atlas_to_reference(
            atlas_file, out_file, xfm_file, ref_file, lut_file=lut_file
        )
        res[f"Subcortrcal_{space}"], self.volume[f"Subcortrcal_{space}"] = out_file, out_file
        # Copy reference file to output directory
        shutil.copy(ref_file, self.out_dir.joinpath(ref_file.name))

        return res


class Anatomical(FreeSurferVolume):
    """Surface-based anatomical processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        fs_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        ses_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
                sub_id: SubjectID.
                fs_dir: Subject's FreeSurfer output directory.
                template_dir: Directory contains required template files.
                out_dir: Directory to store output file.
                ses_id: SessionID. Used in the filename prefix. For example,
                    sub-001_ses-01.
                run_id: RunID. Used in the filename prefix. For example,
                    sub-001_run-1.
        """

        super().__init__(sub_id, fs_dir, template_dir, out_dir, ses_id=ses_id, run_id=run_id)
        # Store important result files
        self.cifti = {}

    def run_anatomical_pipeline(
        self,
        xfm_file: PathLike,
        ref_file: PathLike,
        xfm_mni_file: PathLike,
        ref_mni_file: PathLike,
        keep_gifti_native: bool = False,
        keep_gifti_fsLR: bool = False,
        lut_file: Optional[PathLike] = None,
        lut_subcortical_file: Optional[PathLike] = None,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        msm_config_file: Optional[PathLike] = None,
        inflate_extra_scale: Union[float, int] = 1.0,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Runs full surface-based anatomical process pipeline.

        Args:
            xfm_file: An ITK format affine transformation matrix file.
                Usually it is used to adjust the difference between the
                original and FreeSurfer conformed T1w images.
            ref_file: Reference volume file for xfm_file.
            xfm_mni_file: An ITK format nonlinear transformation matrix
                file. It tranforms image in T1w space to MNI152NLin6Asym
                space.
            ref_mni_file: Reference volume file for xfm_mni_file.
            keep_gifti_native: If ture, keep native space GIFTI files.
            keep_gifti_fsLR: If true, keep fsLR space GIFTI files.
            lut_file: Lut file contains label information of
                FreeSurfer's parcellations. It is used to import label
                information to parcellation NIFTI image header.
            lut_subcortical_file: Lut file contains label information of
                FreeSurfer's subcortical segmentation. It is used to
                import label information to parcellation NIFTI image
                header.
            registration_method: Surface-based registration method.
            msm_config_file: MSMSulc configuration file. Only required
                when registration_method is MSMSulc.
            inflate_extra_scale: Extra iteration scaling value. This
                value is used in function calc_inflation_scale to
                calculate the final iteration scaling value.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.
        """

        # Parse inputs
        if lut_file is None:
            lut_file = self.template_dir.joinpath("FreeSurferAllLut.txt")
        if lut_subcortical_file is None:
            lut_subcortical_file = self.template_dir.joinpath(
                "FreeSurferSubcorticalLabelTableLut.txt"
            )
        if msm_config_file is None:
            msm_config_file = self.template_dir.joinpath("MSMSulcStrainFinalconf")
            if not msm_config_file.is_file():
                raise FileNotFoundError("MSMSulc config!")

        print(f"Starting anatomical pipeline: sub-{self.sub_id}!", flush=True)

        res = {}
        # Check common template data in template_dir
        self.check_template_data()
        # Native surface pipeline
        for hemi in ["L", "R"]:
            out = self.run_native_space_pipeline(
                hemi,
                xfm_file=xfm_file,
                registration_method=registration_method,
                msm_config_file=msm_config_file,
                inflate_extra_scale=inflate_extra_scale,
                debug=debug,
            )
            res.update(out)
        # Resample pipeline
        for hemi in ["L", "R"]:
            out = self.run_resample_fsLR_pipeline(
                hemi,
                target_mesh_density=["164k", "32k"],
                registration_method=registration_method,
                inflate_extra_scale=inflate_extra_scale,
            )
            res.update(out)
        # Volume pipeline
        out = self.run_volume_pipeline(
            xfm_file, ref_file, xfm_mni_file, ref_mni_file, lut_file, lut_subcortical_file
        )
        res.update(out)
        # CIFTI file pipeline
        out = self.run_cifti_pipeline(registration_method=registration_method)
        res.update(out)
        # Make spec file for HCP workbench
        _ = self.make_spec_file("fsnative", "fsnative")
        _ = self.make_spec_file("fsLR", "164k")
        _ = self.make_spec_file("fsLR", "32k")
        # Cleanup
        self.remove_unnecessary_file(
            keep_gifti_native=keep_gifti_native, keep_gifti_fsLR=keep_gifti_fsLR
        )

        print(f"\nAnatomical pipeline finished!\n", flush=True)

        return res

    def run_cifti_pipeline(
        self, registration_method: Literal["FS", "MSMSulc"] = "MSMSulc"
    ) -> dict[str, Path]:
        """Runs CIFTI file pipeline.

        Args:
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.
        """

        res = {}
        # Make metric data dscalar CIFTI file
        for space, mesh_den in [("fsnative", "fsnative"), ("fsLR", "164k"), ("fsLR", "32k")]:
            out = self.make_metric_cifti(space, mesh_den, registration_method=registration_method)
            res.update(out)
        # Make registration distortion dscalar CIFTI file
        for space, mesh_den in [("fsnative", "fsnative"), ("fsLR", "164k"), ("fsLR", "32k")]:
            out = self.make_distortion_cifti(
                space, mesh_den, registration_method=registration_method
            )
            res.update(out)
        # Make atlas dlabel CIFTI file
        for space, mesh_den in [("fsnative", "fsnative"), ("fsLR", "164k"), ("fsLR", "32k")]:
            out = self.make_label_cifti(space, mesh_den, registration_method=registration_method)
            res.update(out)

        return res

    def make_metric_cifti(
        self,
        space: Literal["fsLR", "fsnative"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Makes metric dense scalar CIFTI file.

        Args:
            space: Surface space.
            mesh_den: Surface mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: mesh_den is not fsnative when space is fsnative.
        """

        # Parse surface space
        space = parse_space(space, valid_list=["fsLR", "fsnative"])
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den)
        if space == "fsnative" and mesh_den != "fsnative":
            raise ValueError("If surface space is fsnative, mesh_den should be fsnative as well")
        # Parse registration method
        method = parse_registration_method(registration_method)
        # Addition filename modifier
        desc = "" if space == "fsnative" else f"_desc-{method}"

        print(
            f"\n###Make metric dscalar CIFTI file (space-{space}_den-{mesh_den})###\n", flush=True
        )
        res = {}
        for metric_id in ["sulc", "curv", "thickness"]:

            # Required files
            left_surf_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-L_space-{space}_den-{mesh_den}{desc}_{metric_id}.shape.gii"
            )
            right_surf_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_hemi-R_space-{space}_den-{mesh_den}{desc}_{metric_id}.shape.gii"
            )
            if space == "fsnative":
                left_roi_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-L_space-fsnative_den-fsnative_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
                right_roi_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-R_space-fsnative_den-fsnative_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
            else:
                left_roi_file = self.template_dir.joinpath(
                    f"{space}_hemi-L_space-{space}_den-{mesh_den}_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
                right_roi_file = self.template_dir.joinpath(
                    f"{space}_hemi-R_space-{space}_den-{mesh_den}_"
                    "desc-nomedialwall_probseg.shape.gii"
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_space-{space}_den-{mesh_den}{desc}_{metric_id}.dscalar.nii"
            )
            cifti_map_name = f"{self.anat_prefix}_space-{space}_den-{mesh_den}{desc}_{metric_id}"

            # Make dense scalar
            if metric_id == "sulc":
                out_file = make_dense_scalar(
                    out_file,
                    left_surf_file=left_surf_file,
                    right_surf_file=right_surf_file,
                    cifti_map_name=cifti_map_name,
                )
            else:
                out_file = make_dense_scalar(
                    out_file,
                    left_surf_file=left_surf_file,
                    right_surf_file=right_surf_file,
                    left_roi_file=left_roi_file,
                    right_roi_file=right_roi_file,
                    cifti_map_name=cifti_map_name,
                )

            # Set map palette
            if metric_id == "thickness":
                run_cmd(
                    f"wb_command -disable-provenance -cifti-palette {out_file} "
                    f"MODE_AUTO_SCALE_PERCENTAGE {out_file} -pos-percent 4 96 -interpolate true "
                    "-palette-name videen_style -disp-pos true -disp-neg false -disp-zero false"
                )
            else:
                run_cmd(
                    f"wb_command -disable-provenance -cifti-palette {out_file} "
                    f"MODE_AUTO_SCALE_PERCENTAGE {out_file} -pos-percent 2 98 "
                    "-palette-name Gray_Interp -disp-pos true -disp-neg true -disp-zero true"
                )

            # Record result
            if space == "fsnative":
                res[metric_id], self.cifti[metric_id] = out_file, out_file
            else:
                res[f"{space}_{mesh_den}_{metric_id}"] = out_file
                self.cifti[f"{space}_{mesh_den}_{metric_id}"] = out_file

        return res

    def make_distortion_cifti(
        self,
        space: Literal["fsLR", "fsnative"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Makes registration distortion dense scalar CIFTI file.

        Args:
            space: Surface space.
            mesh_den: Surface mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: mesh_den is not fsnative when space is fsnative.
        """

        # Parse surface space
        space = parse_space(space, valid_list=["fsLR", "fsnative"])
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den)
        if space == "fsnative" and mesh_den != "fsnative":
            raise ValueError("If surface space is fsnative, mesh_den should be fsnative as well")
        # Parse registration method
        # also make FS method registration distortion file, if the main method is not
        method = parse_registration_method(registration_method)
        method_list = [method, "FS"] if method != "FS" else [method]

        print(
            "\n###Make registration distortion dscalar CIFTI file "
            f"(space-{space}_den-{mesh_den})###\n",
            flush=True,
        )
        res = {}
        for method in method_list:
            for metric_id in ["Areal", "Edge", "StrainJ", "StrainR"]:

                # Required files
                left_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-L_space-{space}_den-{mesh_den}_"
                    f"desc-{method}-{metric_id}_distortion.shape.gii"
                )
                right_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-R_space-{space}_den-{mesh_den}_"
                    f"desc-{method}-{metric_id}_distortion.shape.gii"
                )

                # Output
                out_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_space-{space}_den-{mesh_den}_"
                    f"desc-{method}-{metric_id}_distortion.dscalar.nii"
                )
                cifti_map_name = (
                    f"{self.anat_prefix}_space-{space}_den-{mesh_den}_desc-{method}-{metric_id}"
                )

                # Make dense scalar
                out_file = make_dense_scalar(
                    out_file,
                    left_surf_file=left_surf_file,
                    right_surf_file=right_surf_file,
                    cifti_map_name=cifti_map_name,
                )

                # Set map palette
                run_cmd(
                    f"wb_command -disable-provenance -cifti-palette {out_file} "
                    f"MODE_USER_SCALE {out_file} -pos-user 0 1 -neg-user 0 -1 -interpolate true "
                    "-palette-name ROY-BIG-BL -disp-pos true -disp-neg true -disp-zero false"
                )

                # Record result
                res[f"{space}_{mesh_den}_{method}-{metric_id}"] = out_file
                self.cifti[f"{space}_{mesh_den}_{method}-{metric_id}"] = out_file

        return res

    def make_label_cifti(
        self,
        space: Literal["fsLR", "fsnative"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Makes FreeSurfer's annotation dense label CIFTI file.

        Args:
            space: Surface space.
            mesh_den: Surface mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: mesh_den is not fsnative when space is fsnative.
        """

        # Parse surface space
        space = parse_space(space, valid_list=["fsLR", "fsnative"])
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den)
        if space == "fsnative" and mesh_den != "fsnative":
            raise ValueError("If surface space is fsnative, mesh_den should be fsnative as well")
        # Parse registration method
        method = parse_registration_method(registration_method)
        # Addition filename modifier
        desc = "_desc" if space == "fsnative" else f"_desc-{method}"

        print(f"\n###Make atlas dlabel CIFTI file (space-{space}_den-{mesh_den})###\n", flush=True)
        res = {}
        for atlas_id in ["Aparc", "Destrieux", "DKT"]:

            # Required files
            if space == "fsnative":
                left_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-L_space-{space}_den-{mesh_den}_"
                    f"desc-{atlas_id}_dseg.label.gii"
                )
                right_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-R_space-{space}_den-{mesh_den}_"
                    f"desc-{atlas_id}_dseg.label.gii"
                )
                left_roi_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-L_space-fsnative_den-fsnative_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
                right_roi_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-R_space-fsnative_den-fsnative_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
            else:
                left_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-L_space-{space}_den-{mesh_den}_"
                    f"desc-{method}-{atlas_id}_dseg.label.gii"
                )
                right_surf_file = self.out_dir.joinpath(
                    f"{self.anat_prefix}_hemi-R_space-{space}_den-{mesh_den}_"
                    f"desc-{method}-{atlas_id}_dseg.label.gii"
                )
                left_roi_file = self.template_dir.joinpath(
                    f"{space}_hemi-L_space-{space}_den-{mesh_den}_"
                    "desc-nomedialwall_probseg.shape.gii"
                )
                right_roi_file = self.template_dir.joinpath(
                    f"{space}_hemi-R_space-{space}_den-{mesh_den}_"
                    "desc-nomedialwall_probseg.shape.gii"
                )

            # Output
            out_file = self.out_dir.joinpath(
                f"{self.anat_prefix}_space-{space}_den-{mesh_den}{desc}-{atlas_id}_dseg.dlabel.nii"
            )
            cifti_map_name = f"{self.anat_prefix}_space-{space}_den-{mesh_den}{desc}-{atlas_id}"

            # Make dense label
            out_file = make_dense_label(
                out_file,
                left_surf_file=left_surf_file,
                right_surf_file=right_surf_file,
                left_roi_file=left_roi_file,
                right_roi_file=right_roi_file,
                cifti_map_name=cifti_map_name,
            )

            # Record result
            if space == "fsnative":
                res[atlas_id], self.cifti[atlas_id] = out_file, out_file
            else:
                res[f"{space}_{mesh_den}_{method}-{atlas_id}"] = out_file
                self.cifti[f"{space}_{mesh_den}_{method}-{atlas_id}"] = out_file

        return res

    def make_spec_file(
        self,
        space: Literal["fsLR", "fsnative"],
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> Path:
        """Makes spec file for HCP Workbench.

        Args:
            space: Surface space.
            mesh_den: Surface mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A HCP workbench spec file.
        """

        # Parse surface space
        space = parse_space(space, valid_list=["fsLR", "fsnative"])
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den)
        if space == "fsnative" and mesh_den != "fsnative":
            raise ValueError("If surface space is fsnative, mesh_den should be fsnative as well")

        # Output
        spec_file = self.out_dir.joinpath(
            f"{self.anat_prefix}_space-{space}_den-{mesh_den}.wb.spec"
        )
        spec_file.unlink(missing_ok=True)

        print(f"\n###Create spec file (space-{space}_den-{mesh_den})###\n", flush=True)

        # Surface and (no)medialwall ROI
        for hemi, structure in [("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT")]:
            file_list = []
            file_list += sorted(
                self.out_dir.glob(f"sub-*_hemi-{hemi}_space-{space}_den-{mesh_den}*.surf.gii")
            )
            file_list += list(
                self.out_dir.glob(
                    f"sub-*_hemi-{hemi}_space-{space}_den-{mesh_den}*probseg.shape.gii"
                )
            )
            # exclude surfaces used FS registration method if it's not the main method
            if registration_method != "FS":
                file_list = [f for f in file_list if "desc-FS" not in f.name]
            for f in file_list:
                run_cmd(f"wb_command -add-to-spec-file {spec_file} {structure} {f}")
        # Metric data, atlas
        file_list = []
        for metric_id in ["sulc", "curv", "thickness"]:
            file_list += list(
                self.out_dir.glob(f"sub-*_space-{space}_den-{mesh_den}*{metric_id}.dscalar.nii")
            )
        file_list += sorted(
            self.out_dir.glob(f"sub-*_space-{space}_den-{mesh_den}*_dseg.dlabel.nii")
        )
        for f in file_list:
            run_cmd(f"wb_command -add-to-spec-file {spec_file} INVALID {f}")

        if space == "fsLR":
            for hemi, structure in [("L", "CORTEX_LEFT"), ("R", "CORTEX_RIGHT")]:
                file_list = []
                # Template surface
                file_list += sorted(
                    self.template_dir.glob(
                        f"fsLR_hemi-{hemi}_space-{space}_den-{mesh_den}*.surf.gii"
                    )
                )
                # Nomedialwall ROI
                file_list += list(
                    self.template_dir.glob(
                        f"fsLR_hemi-{hemi}_space-{space}_den-{mesh_den}*probseg.shape.gii"
                    )
                )
                for f in file_list:
                    run_cmd(f"wb_command -add-to-spec-file {spec_file} {structure} {f}")
            # Atlas (only 32k)
            if mesh_den == "32k":
                file_list = []
                file_list += sorted(
                    self.template_dir.glob(f"fsLR_space-fsLR_den-32k*_dseg.dlabel.nii")
                )
                for f in file_list:
                    run_cmd(f"wb_command -add-to-spec-file {spec_file} INVALID {f}")
            # Volume
            file_list = []
            file_list += sorted(self.out_dir.glob("sub-*_space-MNI152NLin6Asym*.nii.gz"))
            file_list += sorted(self.out_dir.glob("sub-*_space-T1w*_T1w.nii.gz"))
            for f in file_list:
                run_cmd(f"wb_command -add-to-spec-file {spec_file} INVALID {f}")
        else:
            # Volume
            file_list = []
            file_list += sorted(self.out_dir.glob("sub-*_space-T1w*.nii.gz"))
            for f in file_list:
                run_cmd(f"wb_command -add-to-spec-file {spec_file} INVALID {f}")

        return spec_file

    def remove_unnecessary_file(
        self, keep_gifti_native: bool = False, keep_gifti_fsLR: bool = False
    ) -> None:
        """Removes intermediate files.

        Args:
            keep_gifti_native: If ture, keep native space GIFTI files.
            keep_gifti_fsLR: If true, keep fsLR space GIFTI files.
        """

        if not (keep_gifti_native and keep_gifti_fsLR):
            print("\n###Cleanup unnecessary file###\n", flush=True)

        # fsnative
        if not keep_gifti_native:
            file_list = []
            for _, f in {**self.native["L"], **self.native["R"]}.items():
                # metric
                if ".shape.gii" in Path(f).name:
                    # exclude (no)medialwall ROI
                    if "probseg.shape.gii" not in Path(f).name:
                        file_list.append(f)
                # label
                if "dseg.label.gii" in Path(f).name:
                    file_list.append(f)
            for f in file_list:
                print(f"Cleaning {f.name} ...", flush=True)
                f.unlink()

        # fsLR
        if not keep_gifti_fsLR:
            file_list = []
            for _, f in {**self.fsLR["L"], **self.fsLR["R"]}.items():
                # metric
                if ".shape.gii" in Path(f).name:
                    file_list.append(f)
                # label
                if "dseg.label.gii" in Path(f).name:
                    file_list.append(f)
            for f in file_list:
                print(f"Cleaning {f.name} ...", flush=True)
                f.unlink()


class NativeSurfaceFunc:
    """Native space surface-based functional processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        surf_dir: PathLike,
        out_dir: PathLike,
        anat_ses_id: Optional[str] = None,
        anat_run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
            sub_id: SubjectID.
            surf_dir: Surface files directory. Usually it is the output
                directory of the Anatomical pipeline.
            out_dir: Directory to store output file.
            anat_ses_id: Anatomical image SessionID. It is used for
                selecting surfaces generated by Anatomical pipeline.
            anat_run_id: Anatomical image RunID. It is used for
                selecting surfaces generated by Anatomical pipeline.
        """

        #############
        # Directories
        #############
        self.sub_id = conform_sub_id(sub_id, with_prefix=False)
        self.surf_dir = Path(surf_dir)
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(exist_ok=True, parents=True)
        # regex for BIDS complaint functional file
        self.func_regex = r"sub-[^\W_]+_(ses-[^\W_]+)?_?task-[^\W_]+_(run-\d+)?_?"
        # Anatomical filename prefix
        self.anat_prefix = f"sub-{self.sub_id}"
        if anat_ses_id:
            self.anat_prefix += f"_ses-{anat_ses_id}"
        if anat_run_id:
            self.anat_prefix += f"_run-{anat_run_id}"
        # Store important result files
        self.volume = {}
        self.native = {"L": {}, "R": {}}

    def run_native_space_func_pipeline(
        self,
        func_file: PathLike,
        timestep: Union[float, int],
        ref_file: PathLike,
        smoothing_fwhm: Optional[list[Union[float, int]]] = None,
        timestep_format: str = ":.1f",
        grey_ribbon_value: int = 1,
        neighborhood_smoothing: Union[float, int] = 5,
        ci_limit: Union[float, int] = 0.5,
        dilate_distance: Optional[Union[float, int]] = 10,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Runs native space surface-based functional data pipeline.

        Args:
            func_file: Functional image file.
            timestep: The temporal interval of consecutive time points
                in the func_file. Usually it's the repetition time of
                the functional image.
            ref_file: Volume image file used as reference of generated
                cortical ribbon file.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).
            timestep_format: Float number format in the map name.
            grey_ribbon_value: Index value of the ribbon voxels. See
                function 'make_cortical_ribbon'.
            neighborhood_smoothing: Spatial smoothing kernal sigma
                (FWHM, mm) for finding good voxels. See function
                'make_good_voxel_mask'.
            ci_limit: Parameter to control the good voxel threshold.
                Smaller value relates to stricter threshold. See
                function 'make_good_voxel_mask'.
            dilate_distance: Dilate distance (mm) applies to surface
                sampled functional data.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_file name is not BIDS-compliant.
        """

        # Parse filename prefix
        func_prefix = re.search(self.func_regex, Path(func_file).name)
        if func_prefix is None:
            raise ValueError("The func_file name is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse smoothing fwhm
        smoothing_fwhm = parse_smoothing_fwhm(smoothing_fwhm)

        res = {}
        # Make cortical ribbon mask
        print("\n###Make cortical ribbon mask###\n", flush=True)
        out = self.make_cortical_ribbon(ref_file, grey_ribbon_value=grey_ribbon_value, debug=debug)
        res.update(out)
        # Make good voxel mask
        print("\n###Make good voxel mask###\n", flush=True)
        out = self.make_good_voxel_mask(
            func_file,
            neighborhood_smoothing=neighborhood_smoothing,
            ci_limit=ci_limit,
            debug=debug,
        )
        res.update(out)
        # Sample functional data to surface
        print("\n###Sample functional data to native surface###\n", flush=True)
        for hemi in ["L", "R"]:
            out = self.sample_func_to_surface(
                hemi,
                func_file,
                timestep,
                dilate_distance=dilate_distance,
                timestep_format=timestep_format,
            )
            res.update(out)
        # Smoothing functional data in native space
        if smoothing_fwhm is not None:
            print("\n###Smooth functional data on native surface###\n", flush=True)
            for fwhm in smoothing_fwhm:
                for hemi in ["L", "R"]:
                    out = self.smooth_native_func(hemi, func_prefix, fwhm)
                    res.update(out)
        # Make diagnositic metric
        print("\n###Make diagnositic metric on native surface###\n", flush=True)
        for hemi in ["L", "R"]:
            out = self.make_diagnositic_metric(hemi, func_file, dilate_distance=dilate_distance)
            res.update(out)

        return res

    def make_cortical_ribbon(
        self, ref_file: PathLike, grey_ribbon_value: int = 1, debug: bool = False
    ) -> dict[str, Path]:
        """Make cortical ribbon volume from white and pial surface.

        Args:
            ref_file: Volume image file used as reference of generated
                cortical ribbon file.
            grey_ribbon_value: Index value of the ribbon voxels.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The ref_file name is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse filename prefix
        volume_prefix = re.search(
            self.func_regex + "space-[^\W_]+(_res-[^\W_]+)?_?", Path(ref_file).name
        )
        if volume_prefix is None:
            raise ValueError("The ref_file name is not BIDS-compliant.")
        volume_prefix = volume_prefix.group()
        func_prefix = re.search(self.func_regex, Path(ref_file).name).group()

        # Required file
        left_wm_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-L_space-fsnative_den-fsnative_wm.surf.gii"
        )
        left_pial_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-L_space-fsnative_den-fsnative_pial.surf.gii"
        )
        right_wm_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-R_space-fsnative_den-fsnative_wm.surf.gii"
        )
        right_pial_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-R_space-fsnative_den-fsnative_pial.surf.gii"
        )
        for f in [left_wm_file, left_pial_file, right_wm_file, right_pial_file]:
            if not f.is_file():
                raise FileNotFoundError(f"Surface {f} not found. Run pipeline 'Anatomical' first.")

        # Output
        out_file = self.out_dir.joinpath(volume_prefix + "desc-ribbon_mask.nii.gz")

        # Make ribbon volume
        res = {}
        print(f"Creating cortical ribbon: {ref_file} ...", flush=True)
        out_file = make_cortical_ribbon(
            ref_file,
            out_file,
            left_wm_file=left_wm_file,
            left_pial_file=left_pial_file,
            right_wm_file=right_wm_file,
            right_pial_file=right_pial_file,
            grey_ribbon_value=grey_ribbon_value,
            debug=debug,
        )
        res[f"{func_prefix}ribbon_mask"] = out_file
        self.volume[f"{func_prefix}ribbon_mask"] = out_file

        return res

    def make_good_voxel_mask(
        self,
        func_file: PathLike,
        neighborhood_smoothing: Union[float, int] = 5,
        ci_limit: Union[float, int] = 0.5,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Makes a mask of good cortical voxels.

        Args:
            func_file: Functional image file.
            neighborhood_smoothing: Spatial smoothing kernal sigma (mm).
            ci_limit: Parameter to control the good voxel threshold. Smaller
                value relates to stricter threshold.
            debug: If true, output intermediate files to out_dir.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_file name is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse filename prefix
        volume_prefix = re.search(
            self.func_regex + "space-[^\W_]+(_res-[^\W_]+)?_?", Path(func_file).name
        )
        if volume_prefix is None:
            raise ValueError("The func_file name is not BIDS-compliant.")
        volume_prefix = volume_prefix.group()
        func_prefix = re.search(self.func_regex, Path(func_file).name).group()

        # Required file
        ribbon_file = self.out_dir.joinpath(volume_prefix + "desc-ribbon_mask.nii.gz")
        if not ribbon_file.is_file():
            raise FileNotFoundError(
                f"Cortical ribbon mask {ribbon_file} not found. "
                "Run function 'make_cortical_ribbon' first."
            )

        # Output
        out_file = self.out_dir.joinpath(volume_prefix + "desc-goodvoxel_mask.nii.gz")

        # Find good voxel
        res = {}
        print(f"Creating good voxel mask: {func_file} ...", flush=True)
        out_file = find_good_voxel(
            func_file,
            ribbon_file,
            out_file,
            neighborhood_smoothing=neighborhood_smoothing,
            ci_limit=ci_limit,
            debug=debug,
        )
        res[f"{func_prefix}goodvoxel_mask"] = out_file
        self.volume[f"{func_prefix}goodvoxel_mask"] = out_file

        return res

    def sample_func_to_surface(
        self,
        hemi: Literal["L", "R"],
        func_file: PathLike,
        timestep: Union[float, int],
        dilate_distance: Optional[Union[float, int]] = 10,
        timestep_format: str = ":.1f",
    ) -> dict[str, Path]:
        """Samples volumetric functional data to cortical surface.

        Args:
            hemi: Brain hemisphere.
            func_file: Functional image file.
            timestep: The temporal interval of consecutive time points
                in the func_file. Usually it's the repetition time of
                the functional image.
            dilate_distance: Dilate distance (mm) applies to surface
                sampled functional data.
            timestep_format: Float number format in the map name.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_file name is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse filename prefix
        volume_prefix = re.search(
            self.func_regex + "space-[^\W_]+(_res-[^\W_]+)?_?", Path(func_file).name
        )
        if volume_prefix is None:
            raise ValueError("The func_file name is not BIDS-compliant.")
        volume_prefix = volume_prefix.group()
        func_prefix = re.search(self.func_regex, Path(func_file).name).group()

        # Required file
        wm_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_wm.surf.gii"
        )
        pial_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_pial.surf.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        surf_mask_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        vol_mask_file = self.out_dir.joinpath(volume_prefix + "desc-goodvoxel_mask.nii.gz")
        for f in [wm_file, pial_file, midthickness_file, surf_mask_file]:
            if not f.is_file():
                raise FileNotFoundError(f"Surface {f} not found. Run pipeline 'Anatomical' first.")
        if not vol_mask_file.is_file():
            raise FileNotFoundError(
                f"Good voxel volume mask {vol_mask_file} not found. "
                "Run function 'make_good_voxel_mask' first."
            )

        # Output
        out_file = self.out_dir.joinpath(
            func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_desc-sm0pt0_bold.func.gii"
        )

        # Sample func data to surface
        res = {}
        print(f"Sampling func data to hemi-{hemi} surface: {func_file} ...", flush=True)
        out_file = sample_volume_to_surface(
            func_file,
            wm_file,
            pial_file,
            midthickness_file,
            out_file,
            vol_mask_file=vol_mask_file,
            surf_mask_file=surf_mask_file,
            dilate_distance=dilate_distance,
        )
        # Set GIFTI metadata
        with tempfile.TemporaryDirectory() as tmp_dir:
            name_file = make_func_map_name(
                func_file,
                timestep,
                Path(tmp_dir).joinpath("mapname.txt"),
                float_format=timestep_format,
            )
            run_cmd(
                f"wb_command -disable-provenance -set-map-names {out_file} -name-file {name_file}"
            )
        res[f"{func_prefix}hemi-{hemi}_sm0pt0_bold"] = out_file
        self.native[hemi][f"{func_prefix}hemi-{hemi}_sm0pt0_bold"] = out_file

        return res

    def smooth_native_func(
        self,
        hemi: Literal["L", "R"],
        func_prefix: str,
        smoothing_fwhm: Union[float, int],
    ) -> dict[str, Path]:
        """Smooths native surface functional data.

        Args:
            hemi: Brain hemisphere.
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse smoothing fwhm
        fwhm = convert_fwhm_to_str(smoothing_fwhm)
        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()

        # Required files
        func_file = self.out_dir.joinpath(
            func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_desc-sm0pt0_bold.func.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        surf_mask_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        if not func_file.is_file():
            raise FileNotFoundError(
                f"Native surface functional file {func_file} is not found. "
                "Run function 'sample_func_to_surface' first."
            )
        for f in [midthickness_file, surf_mask_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface {f} is not found. Run pipeline 'Anatomical' first."
                )

        # Output
        out_file = self.out_dir.joinpath(func_file.name.replace("sm0pt0", f"sm{fwhm}"))

        # Smoothing
        print(f"Smoothing func data with FWHM={smoothing_fwhm}mm: {func_file} ...", flush=True)
        res = {}
        out_file = smooth_metric(
            func_file, midthickness_file, out_file, smoothing_fwhm, roi_file=surf_mask_file
        )
        res[f"{func_prefix}hemi-{hemi}_sm{fwhm}_bold"] = out_file
        self.native[hemi][f"{func_prefix}hemi-{hemi}_sm{fwhm}_bold"] = out_file

        return res

    def make_diagnositic_metric(
        self,
        hemi: Literal["L", "R"],
        func_file: PathLike,
        dilate_distance: Optional[Union[float, int]] = 10,
    ) -> dict[str, Path]:
        """Makes diagnositic metric for surface sampled functional data.

        Args:
            hemi: Brain hemisphere.
            func_file: Functional image file.
            dilate_distance: Dilate distance (mm) applies to surface
                sampled functional data.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_file name is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse filename prefix
        volume_prefix = re.search(
            self.func_regex + "space-[^\W_]+(_res-[^\W_]+)?_?", Path(func_file).name
        )
        if volume_prefix is None:
            raise ValueError("The func_file name is not BIDS-compliant.")
        volume_prefix = volume_prefix.group()
        func_prefix = re.search(self.func_regex, Path(func_file).name).group()

        # Required file
        wm_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_wm.surf.gii"
        )
        pial_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_pial.surf.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        surf_mask_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        vol_mask_file = self.out_dir.joinpath(volume_prefix + "desc-goodvoxel_mask.nii.gz")
        for f in [wm_file, pial_file, midthickness_file, surf_mask_file]:
            if not f.is_file():
                raise FileNotFoundError(f"Surface {f} not found. Run pipeline 'Anatomical' first.")
        if not vol_mask_file.is_file():
            raise FileNotFoundError(
                f"Good voxel volume mask {vol_mask_file} not found. "
                "Run function 'make_good_voxel_mask' first."
            )

        print(f"Making diagnositic metric for hemi-{hemi} surface: {func_file} ...", flush=True)
        res = {}
        with tempfile.TemporaryDirectory() as tmp_dir:

            # Calculate temporal mean and coefficient of variation (cov)
            tmean_file = Path(tmp_dir, "Mean.nii.gz")
            tstd_file = Path(tmp_dir, "SD.nii.gz")
            cov_file = Path(tmp_dir, "cov.nii.gz")
            run_cmd(f"fslmaths {func_file} -Tmean {tmean_file} -odt float")
            run_cmd(f"fslmaths {func_file} -Tstd {tstd_file} -odt float")
            run_cmd(f"fslmaths {tstd_file} -div {tmean_file} {cov_file}")

            # Sample Tmean and Cov to surface (only good voxels)
            for metric_id, metric_file in [("mean", tmean_file), ("cov", cov_file)]:
                out_file = Path(self.out_dir).joinpath(
                    func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_"
                    f"desc-goodvoxel_{metric_id}.shape.gii"
                )
                out_file = sample_volume_to_surface(
                    metric_file,
                    wm_file,
                    pial_file,
                    midthickness_file,
                    out_file,
                    vol_mask_file=vol_mask_file,
                    surf_mask_file=surf_mask_file,
                    dilate_distance=dilate_distance,
                )
                run_cmd(
                    f"wb_command -disable-provenance -set-map-names {out_file} "
                    f"-map 1 {func_prefix}hemi-{hemi}_desc-goodvoxel_{metric_id}"
                )
                res[f"{func_prefix}hemi-{hemi}_desc-goodvoxel_{metric_id}"] = out_file
                self.native[hemi][
                    f"{func_prefix}hemi-{hemi}_desc-goodvoxel_{metric_id}"
                ] = out_file

            # Sample Tmean and Cov to surface (all voxels)
            for metric_id, metric_file in [("mean", tmean_file), ("cov", cov_file)]:
                out_file = Path(self.out_dir).joinpath(
                    func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_"
                    f"desc-allvoxel_{metric_id}.shape.gii"
                )
                out_file = sample_volume_to_surface(
                    metric_file,
                    wm_file,
                    pial_file,
                    midthickness_file,
                    out_file,
                    vol_mask_file=None,
                    surf_mask_file=surf_mask_file,
                    dilate_distance=None,
                )
                run_cmd(
                    f"wb_command -disable-provenance -set-map-names {out_file} "
                    f"-map 1 {func_prefix}hemi-{hemi}_desc-allvoxel_{metric_id}"
                )
                res[f"{func_prefix}hemi-{hemi}_desc-allvoxel_{metric_id}"] = out_file
                self.native[hemi][f"{func_prefix}hemi-{hemi}_desc-allvoxel_{metric_id}"] = out_file

            # Sample good voxel mask to surface
            out_file = Path(self.out_dir).joinpath(
                func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_"
                f"desc-goodvoxel_probseg.shape.gii"
            )
            out_file = sample_volume_to_surface(
                vol_mask_file,
                wm_file,
                pial_file,
                midthickness_file,
                out_file,
                vol_mask_file=None,
                surf_mask_file=surf_mask_file,
                dilate_distance=None,
            )
            run_cmd(
                f"wb_command -disable-provenance -set-map-names {out_file} "
                f"-map 1 {func_prefix}hemi-{hemi}_desc-goodvoxel_probseg"
            )
            res[f"{func_prefix}hemi-{hemi}_desc-goodvoxel_probseg"] = out_file
            self.native[hemi][f"{func_prefix}hemi-{hemi}_desc-goodvoxel_probseg"] = out_file

        return res


class ResampleSurfaceFunc(NativeSurfaceFunc):
    """Native surface space functional to fsLR processing pipeline."""

    def __init__(
        self,
        sub_id: Union[int, str],
        surf_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        anat_ses_id: Optional[str] = None,
        anat_run_id: Optional[str] = None,
    ) -> None:
        """Initializes class.

        Args:
            sub_id: SubjectID.
            surf_dir: Surface files directory. Usually it is the output
                directory of the Anatomical pipeline.
            template_dir: Directory contains required template files.
            out_dir: Directory to store output file.
            anat_ses_id: Anatomical image SessionID. It is used for
                selecting surfaces generated by Anatomical pipeline.
            anat_run_id: Anatomical image RunID. It is used for
                selecting surfaces generated by Anatomical pipeline.
        """

        super().__init__(
            sub_id, surf_dir, out_dir, anat_ses_id=anat_ses_id, anat_run_id=anat_run_id
        )
        self.template_dir = Path(template_dir)
        # Store important result files
        self.fsLR = {"L": {}, "R": {}}
        self.cifti = {}

    def run_resample_func_pipeline(
        self,
        func_file: PathLike,
        timestep: Union[float, int],
        mesh_den: str,
        func_std_file: Optional[PathLike] = None,
        smoothing_fwhm: Optional[Union[float, int, list[Union[float, int]]]] = None,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
        debug: bool = False,
    ) -> dict[str, Path]:
        """Runs functional data resample (fsLR, MNI space) pipeline.

        Args:
            func_file: Functional image file. Only used for getting
                filename prefix to match sampled surface files.
            timestep: The temporal interval of consecutive time points
                in the func_file. Usually it's the repetition time of
                the functional image.
            mesh_den: Target fsLR space mesh density.
            func_std_file: Functional image file. It should be in
                MNI152NLin6Asym space with 2mm resolution. Optional.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm). If
                None, no spatial smoothing applies to the functional
                data. It could be a list of numbers indicate multiple
                smoothing levels. The unsmooth data is always generated
                even 0 is not in the smoothing_fwhm list.
            registration_method: Surface-based registration method.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: fun_std_file/func_prefix is not BIDS-compliant.
            ValueError: func_file and func_std_file does not match.
            FileNotFoundError: Required file is not found.
        """

        # Parse functional file name prefix
        func_prefix = re.search(self.func_regex, Path(func_file).name)
        if func_prefix is None:
            raise ValueError("The func_file name is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse smoothing fwhm
        # Note: if 0 presents in the list, remove it from comformed list
        smoothing_fwhm = parse_smoothing_fwhm(smoothing_fwhm, remove_zero=True)
        # Parse func_std_file
        if func_std_file is not None:
            if not Path(func_std_file).name.startswith(func_prefix):
                raise ValueError("Input func_file and func_std_file does not match.")
            with_volume = True
        else:
            with_volume = False

        res = {}
        # Resample functional data from native to fsLR space
        print("\n###Resample functional data to fsLR space###\n", flush=True)
        for hemi in ["L", "R"]:
            out = self.resample_func_to_fsLR(
                hemi,
                func_prefix,
                mesh_den,
                registration_method=registration_method,
            )
            res.update(out)
        # Resample diagnositic metric data from native to fsLR space
        print("\n###Resample diagnositic metric data to fsLR space###\n", flush=True)
        for hemi in ["L", "R"]:
            out = self.resample_diagnositic_metric_to_fsLR(
                hemi,
                func_prefix,
                mesh_den,
                registration_method=registration_method,
            )
            res.update(out)
        # Smoothing functional data in fsLR space
        if smoothing_fwhm is not None:
            print("\n###Smooth functional data in fsLR surface###\n", flush=True)
            for fwhm in smoothing_fwhm:
                for hemi in ["L", "R"]:
                    out = self.smooth_fsLR_func(
                        hemi,
                        func_prefix,
                        mesh_den,
                        fwhm,
                        registration_method=registration_method,
                    )
                    res.update(out)
        # Extract functional data in MNI152NLin6Asym space ROIs
        if with_volume:
            print("\n###Extract functional data in subcortical ROIs###\n", flush=True)
            out = self.extract_func_subcortical(
                func_std_file,
                func_prefix=func_prefix,
                smoothing_fwhm=None,
                debug=debug,
            )
            res.update(out)
            # Smoothing if requested
            if smoothing_fwhm is not None:
                for fwhm in smoothing_fwhm:
                    out = self.extract_func_subcortical(
                        func_std_file,
                        func_prefix=func_prefix,
                        smoothing_fwhm=fwhm,
                        debug=debug,
                    )
                    res.update(out)
        # Make functional dtseries CIFTI file (fsLR, MNI space)
        print("\n###Make functional dtseries CIFTI file###\n", flush=True)
        if func_std_file is None:
            print(
                "Argument func_std_file is None. "
                "Output CIFTI files does not include subcortical volume data.\n",
                flush=True,
            )
        out = self.make_func_fsLR_cifti(
            func_prefix, timestep, mesh_den, smoothing_fwhm=None, include_volume=with_volume
        )
        res.update(out)
        # Smoothed file
        if smoothing_fwhm is not None:
            for fwhm in smoothing_fwhm:
                out = self.make_func_fsLR_cifti(
                    func_prefix,
                    timestep,
                    mesh_den,
                    smoothing_fwhm=fwhm,
                    include_volume=with_volume,
                )
                res.update(out)
        # Make diagnositic dscalar CIFTI file (fsLR)
        print("\n###Make diagnositic dscalar CIFTI file###\n", flush=True)
        out = self.make_diagnositic_metric_fsLR_cifti(func_prefix, mesh_den, smoothing_fwhm=None)
        res.update(out)
        # Smoothed file
        if smoothing_fwhm is not None:
            for fwhm in smoothing_fwhm:
                out = self.make_diagnositic_metric_fsLR_cifti(
                    func_prefix, mesh_den, smoothing_fwhm=fwhm
                )
                res.update(out)

        return res

    def resample_func_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        func_prefix: str,
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native surface functional data to fsLR space.

        Args:
            hemi: Brain hemisphere.
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        func_file = self.out_dir.joinpath(
            func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_desc-sm0pt0_bold.func.gii"
        )
        warpped_sphere_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        warpped_midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        roi_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        template_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        if not func_file.is_file():
            raise FileNotFoundError(
                f"Native surface functional file {func_file} is not found."
                "Run pipeline 'run_native_space_func_pipeline' first."
            )
        for f in [
            warpped_sphere_file,
            template_sphere_file,
            midthickness_file,
            warpped_midthickness_file,
            roi_file,
            template_roi_file,
        ]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface {f} is not found. Run pipeline 'Anatomical' first."
                )

        # Output
        out_file = self.out_dir.joinpath(
            func_prefix + f"hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-sm0pt0_bold.func.gii"
        )

        # Resample func data
        print(f"Resampling func data to fsLR {mesh_den} space: {func_file} ...", flush=True)
        res = {}
        out_file = resample_metric(
            func_file,
            warpped_sphere_file,
            template_sphere_file,
            out_file,
            current_area_surf_file=midthickness_file,
            target_area_surf_file=warpped_midthickness_file,
            roi_file=roi_file,
            resample_method="ADAP_BARY_AREA",
        )
        # Apply (no)medialwall in target space
        run_cmd(
            f"wb_command -disable-provenance -metric-mask {out_file} {template_roi_file} {out_file}"
        )
        res[f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_sm0pt0_bold"] = out_file
        self.fsLR[hemi][f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_sm0pt0_bold"] = out_file

        return res

    def resample_diagnositic_metric_to_fsLR(
        self,
        hemi: Literal["L", "R"],
        func_prefix: str,
        mesh_den: str,
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Resamples native surface diagnositic metric to fsLR space.

        Args:
            hemi: Brain hemisphere.
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            mesh_den: Target fsLR space mesh density.
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse hemisphere
        hemi, _ = parse_hemi(hemi)
        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Common required files
        warpped_sphere_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-fsnative_desc-{method}_sphere.surf.gii"
        )
        template_sphere_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_midthickness.surf.gii"
        )
        warpped_midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        roi_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsnative_den-fsnative_"
            "desc-nomedialwall_probseg.shape.gii"
        )
        template_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        for f in [
            warpped_sphere_file,
            template_sphere_file,
            midthickness_file,
            warpped_midthickness_file,
            roi_file,
            template_roi_file,
        ]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface {f} is not found. Run pipeline 'Anatomical' first."
                )

        res = {}
        for suffix in [
            "desc-goodvoxel_mean",
            "desc-goodvoxel_cov",
            "desc-allvoxel_mean",
            "desc-allvoxel_cov",
            "desc-goodvoxel_probseg",
        ]:
            # Required file
            metric_file = self.out_dir.joinpath(
                func_prefix + f"hemi-{hemi}_space-fsnative_den-fsnative_{suffix}.shape.gii"
            )
            if not metric_file.is_file():
                raise FileNotFoundError(
                    f"Native surface diagnositic metric file {metric_file} is not found."
                    "Run function 'make_diagnositic_metric' first."
                )

            # Output
            out_file = self.out_dir.joinpath(
                func_prefix + f"hemi-{hemi}_space-fsLR_den-{mesh_den}_{suffix}.shape.gii"
            )
            # Resample diagnositic data
            print(
                f"Resampling diagnositic metric data to fsLR {mesh_den} space: {metric_file} ...",
                flush=True,
            )
            out_file = resample_metric(
                metric_file,
                warpped_sphere_file,
                template_sphere_file,
                out_file,
                current_area_surf_file=midthickness_file,
                target_area_surf_file=warpped_midthickness_file,
                roi_file=roi_file,
                resample_method="ADAP_BARY_AREA",
            )
            # Apply (no)medialwall in target space
            run_cmd(
                f"wb_command -disable-provenance -metric-mask {out_file} "
                f"{template_roi_file} {out_file}"
            )
            res[f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_{suffix}"] = out_file
            self.fsLR[hemi][f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_{suffix}"] = out_file

        return res

    def smooth_fsLR_func(
        self,
        hemi: Literal["L", "R"],
        func_prefix: str,
        mesh_den: str,
        smoothing_fwhm: Union[float, int],
        registration_method: Literal["FS", "MSMSulc"] = "MSMSulc",
    ) -> dict[str, Path]:
        """Smooths fsLR space functional data.

        Args:
            hemi: Brain hemisphere.
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            mesh_den: Target fsLR space mesh density.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).
            registration_method: Surface-based registration method.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse smoothing fwhm
        fwhm = convert_fwhm_to_str(smoothing_fwhm)
        # Parse registration method
        method = parse_registration_method(registration_method)

        # Required files
        func_file = self.out_dir.joinpath(
            func_prefix + f"hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-sm0pt0_bold.func.gii"
        )
        midthickness_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_hemi-{hemi}_space-fsLR_den-{mesh_den}_"
            f"desc-{method}_midthickness.surf.gii"
        )
        surf_mask_file = self.template_dir.joinpath(
            f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        if not func_file.is_file():
            raise FileNotFoundError(
                f"fsLR space functional file {func_file} is not found."
                "Run function 'resample_func_to_fsLR' first."
            )
        for f in [midthickness_file, surf_mask_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface {f} is not found. Run pipeline 'Anatomical' first."
                )

        # Output
        out_file = self.out_dir.joinpath(func_file.name.replace("sm0pt0", f"sm{fwhm}"))

        # Smoothing
        print(f"Smoothing func data with FWHM={smoothing_fwhm}mm: {func_file} ...", flush=True)
        res = {}
        out_file = smooth_metric(
            func_file, midthickness_file, out_file, smoothing_fwhm, roi_file=surf_mask_file
        )
        res[f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_sm{fwhm}_bold"] = out_file
        self.fsLR[hemi][f"{func_prefix}hemi-{hemi}_fsLR_{mesh_den}_sm{fwhm}_bold"] = out_file

        return res

    def extract_func_subcortical(
        self,
        func_std_file: PathLike,
        func_prefix: Optional[str] = None,
        smoothing_fwhm: Optional[Union[float, int]] = None,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Extracts functional data in standard subcortical ROIs.

        Args:
            func_std_file: Functional image file. It should be in
                MNI152NLin6Asym space with 2mm resolution.
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm). If
                None, no spatial smoothing applies to the functional
                data. Note, this operation is constrained within each
                subcortical region.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The fun_std_file/func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse func_std_file
        func_std_prefix = re.search(self.func_regex, Path(func_std_file).name)
        if func_std_prefix is None:
            raise ValueError("The func_std_file name is not BIDS-compliant.")
        # Parse func_prefix
        if func_prefix is None:
            func_prefix = re.search(self.func_regex, Path(func_std_file).name)
            if func_prefix is None:
                raise ValueError("The func_std_file name is not BIDS-compliant.")
        else:
            func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
            func_prefix = re.search(self.func_regex, func_prefix)
            if func_prefix is None:
                raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse smoothing fwhm
        if smoothing_fwhm is not None:
            fwhm = smoothing_fwhm
            fwhm_str = convert_fwhm_to_str(smoothing_fwhm)
        else:
            fwhm = 0
            fwhm_str = "0pt0"

        # Required files
        seg_file = self.surf_dir.joinpath(
            f"{self.anat_prefix}_space-MNI152NLin6Asym_res-2_desc-Subcortical_dseg.nii.gz"
        )
        template_seg_file = self.template_dir.joinpath(
            "MNI_space-MNI152NLin6Asym_res-2_desc-Subcortical_dseg.nii.gz"
        )
        for f in [seg_file, template_seg_file]:
            if not f.is_file():
                raise FileNotFoundError(f"ROI {f} is not found. Run pipeline 'Anatomical' first.")

        # Output
        out_file = self.out_dir.joinpath(
            func_prefix + f"space-MNI152NLin6Asym_res-02_desc-Subcortical-sm{fwhm_str}_bold.nii.gz"
        )

        # Extract func data
        print(
            f"Extracting func data in subcortical ROIs with FWHM={fwhm}mm: {func_std_file} ...",
            flush=True,
        )
        res = {}
        out_file = extract_func_subcortical(
            func_std_file,
            seg_file,
            template_seg_file,
            out_file,
            smoothing_fwhm=smoothing_fwhm,
            debug=debug,
        )
        res[f"{func_prefix}MNI152NLin6Asym_res-02_Subcortical_sm{fwhm_str}"] = out_file
        self.volume[f"{func_prefix}MNI152NLin6Asym_res-02_Subcortical_sm{fwhm_str}"] = out_file

        return res

    def make_func_fsLR_cifti(
        self,
        func_prefix: str,
        timestep: Union[float, int],
        mesh_den: str,
        smoothing_fwhm: Optional[Union[float, int]] = None,
        include_volume: bool = True,
    ) -> dict[str, Path]:
        """Makes functional dtseries CIFTI file (fsLR, MNI space).

        Args:
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            timestep: The temporal interval of consecutive time points
                in the func_file. Usually it's the repetition time of
                the functional image.
            mesh_den: Target fsLR space mesh density.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).
            include_volume: If true, include functional data in volume
                standard subcortical ROIs.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix name is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse smoothing fwhm
        if smoothing_fwhm is None:
            smoothing_fwhm = 0
        fwhm_str = convert_fwhm_to_str(smoothing_fwhm)

        # Required file
        left_surf_file = self.out_dir.joinpath(
            func_prefix + f"hemi-L_space-fsLR_den-{mesh_den}_desc-sm{fwhm_str}_bold.func.gii"
        )
        right_surf_file = self.out_dir.joinpath(
            func_prefix + f"hemi-R_space-fsLR_den-{mesh_den}_desc-sm{fwhm_str}_bold.func.gii"
        )
        left_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-L_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        right_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-R_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        for f in [left_surf_file, right_surf_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"Surface functional file {f} is not found. "
                    "Run function 'resample_func_to_fsLR' first."
                )
        for f in [left_roi_file, right_roi_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"ROI file {f} is not found. Run pipeline 'Anatomical' first."
                )
        if include_volume:
            volume_file = self.out_dir.joinpath(
                func_prefix
                + f"space-MNI152NLin6Asym_res-02_desc-Subcortical-sm{fwhm_str}_bold.nii.gz"
            )
            template_seg_file = self.template_dir.joinpath(
                "MNI_space-MNI152NLin6Asym_res-2_desc-Subcortical_dseg.nii.gz"
            )
            if not volume_file.is_file():
                raise FileNotFoundError(
                    f"Functional file {volume_file} is not found. "
                    "Run function 'extract_func_subcortical' first."
                )
            if not template_seg_file.is_file():
                raise FileNotFoundError(
                    f"ROI file {volume_file} is not found. Run pipeline 'Anatomical' first."
                )
        else:
            volume_file, template_seg_file = None, None

        # Output
        out_file = self.out_dir.joinpath(
            func_prefix + f"space-fsLR_den-{mesh_den}_desc-sm{fwhm_str}_bold.dtseries.nii"
        )

        # Make func dtseries
        res = {}
        out_file = make_dense_timeseries(
            out_file,
            timestep,
            left_surf_file=left_surf_file,
            right_surf_file=right_surf_file,
            left_roi_file=left_roi_file,
            right_roi_file=right_roi_file,
            volume_file=volume_file,
            volume_label_file=template_seg_file,
        )
        res[f"{func_prefix}fsLR_32k_sm{fwhm_str}"] = out_file
        self.cifti[f"{func_prefix}fsLR_32k_sm{fwhm_str}"] = out_file

        return res

    def make_diagnositic_metric_fsLR_cifti(
        self,
        func_prefix: str,
        mesh_den: str,
        smoothing_fwhm: Optional[Union[float, int]] = None,
    ) -> dict[str, Path]:
        """Makes diagnositic metric dscalar CIFTI file (fsLR).

        Args:
            func_prefix: Functional image filename prefix. For example,
                sub-001_ses-01_task-XXX_run-1_. It should match the
                surface sampled functional file.
            mesh_den: Target fsLR space mesh density.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm).

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: The func_prefix is not BIDS-compliant.
            FileNotFoundError: Required file is not found.
        """

        # Parse func_prefix
        func_prefix = f"{func_prefix}_" if not func_prefix.endswith("_") else func_prefix
        func_prefix = re.search(self.func_regex, func_prefix)
        if func_prefix is None:
            raise ValueError("The func_prefix is not BIDS-compliant.")
        func_prefix = func_prefix.group()
        # Parse target mesh density
        mesh_den = parse_mesh_density(mesh_den, valid_list=["164k", "59k", "32k"])
        # Parse smoothing fwhm
        if smoothing_fwhm is None:
            smoothing_fwhm = 0
        else:
            print(
                "Only make temporal mean (good voxel) CIFTI file "
                "when smoothing kernal size is not 0.",
                flush=True,
            )
        fwhm_str = convert_fwhm_to_str(smoothing_fwhm)

        # Common required file
        left_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-L_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        right_roi_file = self.template_dir.joinpath(
            f"fsLR_hemi-R_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
        )
        for f in [left_roi_file, right_roi_file]:
            if not f.is_file():
                raise FileNotFoundError(
                    f"ROI file {f} is not found. Run pipeline 'Anatomical' first."
                )

        res = {}

        # Tmean (good voxel)
        func_file = self.out_dir.joinpath(
            func_prefix + f"space-fsLR_den-{mesh_den}_desc-sm{fwhm_str}_bold.dtseries.nii"
        )
        if not func_file.is_file():
            raise FileNotFoundError(
                f"Functional dtseries file {f} is not found. "
                "Run function 'make_func_fsLR_cifti' first."
            )
        # Output
        out_file = self.out_dir.joinpath(
            func_prefix + f"space-fsLR_den-{mesh_den}_desc-goodvoxel-sm{fwhm_str}_mean.dscalar.nii"
        )
        # Make dscalar file
        print(f"Creating dense scalar file: {out_file} ...", flush=True)
        run_cmd(f"wb_command -disable-provenance -cifti-reduce {func_file} MEAN {out_file}")
        run_cmd(
            f"wb_command -disable-provenance -set-map-names {out_file} -map 1 "
            f"{func_prefix}space-fsLR_den-{mesh_den}_desc-goodvoxel-sm{fwhm_str}_mean"
        )
        res[f"{func_prefix}fsLR_32k_desc-sm{fwhm_str}-goodvoxel_mean"] = out_file
        self.cifti[f"{func_prefix}fsLR_32k_desc-sm{fwhm_str}-goodvoxel_mean"] = out_file

        # Make other files when smoothing is 0
        if smoothing_fwhm == 0:
            for suffix in [
                "desc-goodvoxel_cov",
                "desc-allvoxel_mean",
                "desc-allvoxel_cov",
                "desc-goodvoxel_probseg",
            ]:
                # Required file
                left_surf_file = self.out_dir.joinpath(
                    func_prefix + f"hemi-L_space-fsLR_den-{mesh_den}_{suffix}.shape.gii"
                )
                right_surf_file = self.out_dir.joinpath(
                    func_prefix + f"hemi-L_space-fsLR_den-{mesh_den}_{suffix}.shape.gii"
                )
                for f in [left_surf_file, right_surf_file]:
                    if not f.is_file():
                        raise FileNotFoundError(
                            f"Mertic file {f} is not found. "
                            "Run function 'resample_diagnositic_metric_to_fsLR' first."
                        )
                # Output
                out_file = self.out_dir.joinpath(
                    func_prefix + f"space-fsLR_den-{mesh_den}_{suffix}.dscalar.nii"
                )
                # Make dscalar file
                out_file = make_dense_scalar(
                    out_file,
                    left_surf_file=left_surf_file,
                    right_surf_file=right_surf_file,
                    left_roi_file=left_roi_file,
                    right_roi_file=right_roi_file,
                    cifti_map_name=f"{func_prefix}space-fsLR_den-{mesh_den}_{suffix}",
                )
                self.cifti[f"{func_prefix}fsLR_32k_{suffix}"] = out_file

        return res


class FunctionalSurfaceBased(ResampleSurfaceFunc):
    def __init__(
        self,
        sub_id: Union[int, str],
        surf_dir: PathLike,
        template_dir: PathLike,
        out_dir: PathLike,
        anat_ses_id: Optional[str] = None,
        anat_run_id: Optional[str] = None,
    ):
        """Initializes class.

        Args:
            sub_id: SubjectID.
            surf_dir: Surface files directory. Usually it is the output
                directory of the Anatomical pipeline.
            template_dir: Directory contains required template files.
            out_dir: Directory to store output file.
            anat_ses_id: Anatomical image SessionID. It is used for
                selecting surfaces generated by Anatomical pipeline.
            anat_run_id: Anatomical image RunID. It is used for
                selecting surfaces generated by Anatomical pipeline.
        """

        super().__init__(
            sub_id,
            surf_dir,
            template_dir,
            out_dir,
            anat_ses_id=anat_ses_id,
            anat_run_id=anat_run_id,
        )

    def run_functional_pipeline(
        self,
        func_file: PathLike,
        timestep: Union[float, int],
        ref_file: PathLike,
        mesh_den: str,
        func_std_file: Optional[PathLike] = None,
        smoothing_fwhm: Optional[Union[float, int, list[Union[float, int]]]] = None,
        registration_method: str = "MSMSulc",
        keep_gifti_native: bool = False,
        keep_gifti_fsLR: bool = False,
        timestep_format: str = ":.1f",
        grey_ribbon_value: int = 1,
        neighborhood_smoothing: Union[float, int] = 5,
        ci_limit: Union[float, int] = 0.5,
        dilate_distance: Optional[Union[float, int]] = 10,
        debug: bool = False,
    ) -> dict[str, Path]:
        """Runs full surface-based functional data pipeline.

        Args:
            func_file: Functional image file.
            timestep: The temporal interval of consecutive time points
                in the func_file. Usually it's the repetition time of
                the functional image.
            ref_file: Volume image file used as reference of generated
                cortical ribbon file.
            mesh_den: Target fsLR space mesh density.
            func_std_file: Functional image file. It should be in
                MNI152NLin6Asym space with 2mm resolution. Optional.
            smoothing_fwhm: Spatial smoothing kernal size (FWHM, mm). If
                None, no spatial smoothing applies to the functional
                data. It could be a list of numbers indicate multiple
                smoothing levels. The unsmooth data is always generated
                even 0 is not in the smoothing_fwhm list.
            registration_method: Surface-based registration method.
            keep_gifti_native: If ture, keep native space GIFTI files.
            keep_gifti_fsLR: If true, keep fsLR space GIFTI files.
            timestep_format: Float number format in the map name.
            grey_ribbon_value: Index value of the ribbon voxels. See
                function 'make_cortical_ribbon'.
            neighborhood_smoothing: Spatial smoothing kernal sigma
                (FWHM, mm) for finding good voxels. See function
                'make_good_voxel_mask'.
            ci_limit: Parameter to control the good voxel threshold.
                Smaller value relates to stricter threshold. See
                function 'make_good_voxel_mask'.
            dilate_distance: Dilate distance (mm) applies to surface
                sampled functional data.
            debug: If true, output intermediate files.

        Returns:
            A dict stores generated files.

        Raises:
            ValueError: fun_std_file/func_prefix is not BIDS-compliant.
            ValueError: func_file and func_std_file does not match.
            FileNotFoundError: Required file is not found.

        """

        print(f"Starting surface-based functional pipeline: {func_file.name}!", flush=True)

        res = {}
        # Native space functional data pipeline
        smoothing_fwhm_native = smoothing_fwhm if keep_gifti_native else None
        out = self.run_native_space_func_pipeline(
            func_file,
            timestep,
            ref_file,
            smoothing_fwhm=smoothing_fwhm_native,
            timestep_format=timestep_format,
            grey_ribbon_value=grey_ribbon_value,
            neighborhood_smoothing=neighborhood_smoothing,
            ci_limit=ci_limit,
            dilate_distance=dilate_distance,
            debug=debug,
        )
        res.update(out)
        # Resample pipeline
        out = self.run_resample_func_pipeline(
            func_file,
            timestep,
            mesh_den,
            func_std_file=func_std_file,
            smoothing_fwhm=smoothing_fwhm,
            registration_method=registration_method,
            debug=debug,
        )
        res.update(out)
        # Cleanup
        print("\n###Cleanup unnecessary file###\n", flush=True)
        self.remove_unnecessary_file(
            keep_gifti_native=keep_gifti_native, keep_gifti_fsLR=keep_gifti_fsLR
        )

        print(f"\nSurface-based functional pipeline finished!\n\n\n", flush=True)

        return res

    def remove_unnecessary_file(
        self, keep_gifti_native: bool = False, keep_gifti_fsLR: bool = False
    ) -> None:
        """Removes intermediate files.

        Args:
            keep_gifti_native: If ture, keep native space GIFTI files.
            keep_gifti_fsLR: If true, keep fsLR space GIFTI files.
        """

        # fsnative
        if not keep_gifti_native:
            # GIFTI surface
            for _, f in {**self.native["L"], **self.native["R"]}.items():
                print(f"Cleaning {f.name} ...", flush=True)
                f.unlink()
            # Ribbon and good voxel mask
            for _, f in self.volume.items():
                for s in ["ribbon_mask", "goodvoxel_mask"]:
                    if s in f.name:
                        print(f"Cleaning {f.name} ...", flush=True)
                        f.unlink()
        # fsLR file
        if not keep_gifti_fsLR:
            # GIFTI surface
            for _, f in {**self.fsLR["L"], **self.fsLR["R"]}.items():
                print(f"Cleaning {f.name} ...", flush=True)
                f.unlink()
            # Subcortical volume
            for _, f in self.volume.items():
                for s in ["Subcortical"]:
                    if s in f.name:
                        print(f"Cleaning {f.name} ...", flush=True)
                        f.unlink()
