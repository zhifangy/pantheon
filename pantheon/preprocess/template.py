#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Union
from pathlib import Path
import shutil

from ..image.gifti import sanitize_gii_metadata
from ..utils.typing import PathLike


def copy_template_file(
    out_dir: PathLike,
    src_dir: Optional[PathLike] = None,
    check_output_only: bool = False,
) -> dict[str, Union[list[Path], dict[str, list[Path]]]]:
    """Copies standard files from HCP offical release.

    Template folder should be the custom `tpl-HCP_S1200`.

    Args:
        out_dir: Ouput directory to store template files.
        src_dir: Template file directory. If check_ouput_only is true,
            this could be None.
        check_output_only: If true, only check whether required files
            are presented in the out_dir, instead of copying them from
            src_dir.

    Returns:
        A dict contains template files path.

    Raises:
        FileNotFoundError: Template file is not found in out_dir.
    """

    if check_output_only:
        print("Checking standard files from HCP offical release...", flush=True)
    else:
        print("Copying standard files from HCP offical release...", flush=True)

    # Directory
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    if check_output_only:
        src_dir = ""
    mesh_dir = Path(src_dir).joinpath("standard_mesh_atlases")
    atlas_dir = Path(src_dir).joinpath("S1200_Group_Avg_32k")
    lut_dir = Path(src_dir).joinpath("Lut")
    config_dir = Path(src_dir).joinpath("Config")

    # Output file record
    file_dict = {"gifti": {"L": [], "R": []}, "cifti": [], "volume": [], "other": []}

    # Surface file
    copy_list = []
    for hemi in ["L", "R"]:
        # fsLR standard sphere surface (164k, 59k, 32k)
        src_file = mesh_dir.joinpath(f"fsaverage.{hemi}_LR.spherical_std.164k_fs_LR.surf.gii")
        dst_file = out_dir.joinpath(f"fsLR_hemi-{hemi}_space-fsLR_den-164k_sphere.surf.gii")
        copy_list.append((src_file, dst_file))
        file_dict["gifti"][hemi].append(dst_file)
        for mesh_den in ["59k", "32k"]:
            src_file = mesh_dir.joinpath(f"{hemi}.sphere.{mesh_den}_fs_LR.surf.gii")
            dst_file = out_dir.joinpath(
                f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_sphere.surf.gii"
            )
            copy_list.append((src_file, dst_file))
            file_dict["gifti"][hemi].append(dst_file)
        # fsLR standard surface (only 32k)
        for surf_id in ["wm", "pial", "midthickness", "inflated", "veryinflated", "flat"]:
            src_file = atlas_dir.joinpath(
                f"S1200_hemi-{hemi}_space-fsLR_den-32k_desc-MSMAll_{surf_id}.surf.gii"
            )
            dst_file = out_dir.joinpath(f"fsLR_hemi-{hemi}_space-fsLR_den-32k_{surf_id}.surf.gii")
            copy_list.append((src_file, dst_file))
            file_dict["gifti"][hemi].append(dst_file)
        # fsaverage sphere surface (164k) in fsLR space (164k) (for registration)
        src_file = mesh_dir.joinpath(
            f"fs_{hemi}",
            f"fs_{hemi}-to-fs_LR_fsaverage.{hemi}_LR.spherical_std.164k_fs_{hemi}.surf.gii",
        )
        dst_file = out_dir.joinpath(f"fsaverage_hemi-{hemi}_space-fsLR_den-164k_sphere.surf.gii")
        copy_list.append((src_file, dst_file))
        file_dict["gifti"][hemi].append(dst_file)
        # fsaverage standard sphere surface (164k) (for registration)
        src_file = mesh_dir.joinpath(
            f"fs_{hemi}", f"fsaverage.{hemi}.sphere.164k_fs_{hemi}.surf.gii"
        )
        dst_file = out_dir.joinpath(
            f"fsaverage_hemi-{hemi}_space-fsaverage_den-164k_sphere.surf.gii"
        )
        copy_list.append((src_file, dst_file))
        file_dict["gifti"][hemi].append(dst_file)
    for src_file, dst_file in copy_list:
        if check_output_only:
            assert dst_file.is_file(), f"File {dst_file} not found."
        else:
            shutil.copy(src_file, dst_file)
            _ = sanitize_gii_metadata(
                dst_file, dst_file, da_meta={"AnatomicalStructureSecondary": "MidThickness"}
            )

    # Surface metric file
    copy_list = []
    for hemi in ["L", "R"]:
        # fsLR sulc metric (164k, or MSM registration)
        src_file = mesh_dir.joinpath(f"{hemi}.refsulc.164k_fs_LR.shape.gii")
        dst_file = out_dir.joinpath(f"fsLR_hemi-{hemi}_space-fsLR_den-164k_sulc.shape.gii")
        copy_list.append((src_file, dst_file))
        file_dict["gifti"][hemi].append(dst_file)
    for src_file, dst_file in copy_list:
        if check_output_only:
            assert dst_file.is_file(), f"File {dst_file} not found."
        else:
            shutil.copy(src_file, dst_file)
            _ = sanitize_gii_metadata(dst_file, dst_file)

    # Surface ROI file
    copy_list = []
    for hemi in ["L", "R"]:
        # (no)medialwall ROI (164k, 59k, 32k)
        for mesh_den in ["164k", "59k", "32k"]:
            src_file = mesh_dir.joinpath(f"{hemi}.atlasroi.{mesh_den}_fs_LR.shape.gii")
            dst_file = out_dir.joinpath(
                f"fsLR_hemi-{hemi}_space-fsLR_den-{mesh_den}_desc-nomedialwall_probseg.shape.gii"
            )
            copy_list.append((src_file, dst_file))
            file_dict["gifti"][hemi].append(dst_file)
    for src_file, dst_file in copy_list:
        if check_output_only:
            assert dst_file.is_file(), f"File {dst_file} not found."
        else:
            shutil.copy(src_file, dst_file)
            _ = sanitize_gii_metadata(
                dst_file, dst_file, da_meta={"Name": f"fsLR_hemi-{hemi}_desc-nomedialwall"}
            )

    # CIFTI, volume, lut and MSMSulc config file
    copy_list = []
    # Atlas file (32k)
    # MMP1, Brodmann and RSN
    for label_id in ["MMP1", "Brodmann", "RSN"]:
        src_file = atlas_dir.joinpath(f"{label_id}_space-fsLR_den-32k_dseg.dlabel.nii")
        dst_file = out_dir.joinpath(f"{label_id}_space-fsLR_den-32k_dseg.dlabel.nii")
        copy_list.append((src_file, dst_file))
        file_dict["cifti"].append(dst_file)
    # Schaefer2018
    src_file = atlas_dir.joinpath(
        f"Schaefer2018_space-fsLR_den-32k_desc-400Parcels17Networks_dseg.dlabel.nii"
    )
    dst_file = out_dir.joinpath(
        f"Schaefer2018_space-fsLR_den-32k_desc-400Parcels17Networks_dseg.dlabel.nii"
    )
    copy_list.append((src_file, dst_file))
    file_dict["cifti"].append(dst_file)
    # Subcortical volume ROI file in MNI space
    src_file = mesh_dir.joinpath("Atlas_ROIs.2.nii.gz")
    dst_file = out_dir.joinpath("ASeg_space-MNI152NLin6Asym_res-2_desc-Subcortical_dseg.nii.gz")
    copy_list.append((src_file, dst_file))
    file_dict["volume"].append(dst_file)
    # Lut
    for fname in ["FreeSurferAllLut.txt", "FreeSurferSubcorticalLabelTableLut.txt"]:
        src_file = lut_dir.joinpath(fname)
        dst_file = out_dir.joinpath(fname)
        copy_list.append((src_file, dst_file))
        file_dict["other"].append(dst_file)
    # MSMSulc config
    src_file = config_dir.joinpath("MSMSulcStrainFinalconf")
    dst_file = out_dir.joinpath("MSMSulcStrainFinalconf")
    copy_list.append((src_file, dst_file))
    file_dict["other"].append(dst_file)
    for src_file, dst_file in copy_list:
        if check_output_only:
            if not dst_file.is_file():
                raise FileNotFoundError(f"File {dst_file} is not found.")
        else:
            shutil.copy(src_file, dst_file)

    return file_dict
