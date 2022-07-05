#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations
from typing import Optional, Literal
from pathlib import Path
import shutil
import tempfile

from ..image.gifti import sanitize_gii_metadata
from ..utils.validation import parse_hemi
from ..utils.shell import run_cmd
from ..utils.typing import PathLike


def convert_freesurfer_geometry_surface(
    sub_id: str,
    hemi: Literal["L", "R"],
    surf_id: str,
    fs_dir: PathLike,
    out_dir: PathLike,
    adjust_cras: bool = True,
    xfm_file: Optional[PathLike] = None,
    debug: bool = False,
) -> Path:
    """Converts FreeSurfer's geometry surfaces to GIFTI format.

    Args:
        sub_id: SubjectID.
        hemi: Brain hemisphere.
        surf_id: Surface name in FreeSurfer's outputs.
        fs_dir: Subject's FreeSurfer output directory.
        out_dir: Directory to store output file.
        adjust_cras: If true, adjust the cras offset which FreeSurfer
            stores in file's header.
        xfm_file: An ITK format affine transformation matrix file. If it
            is given, applying it to the surface file. Optional.
        debug: If true, output intermediate files to out_dir.

    Returns:
        A surface mesh file in GIFTI format.

    Raises:
        ValueError: Unrecognized surf_id.
    """

    # Parse hemisphere
    hemi, structure = parse_hemi(hemi)

    # Surface metadata
    if surf_id in ["white", "wm"]:
        fs_surf_id, surf_id, surf_type, surf_secondary_type = (
            "white",
            "wm",
            "ANATOMICAL",
            "GRAY_WHITE",
        )
    elif surf_id == "pial":
        fs_surf_id, surf_id, surf_type, surf_secondary_type = (
            "pial",
            "pial",
            "ANATOMICAL",
            "PIAL",
        )
    elif surf_id == "sphere":
        fs_surf_id, surf_id, surf_type, surf_secondary_type = (
            "sphere",
            "sphere",
            "SPHERICAL",
            None,
        )
    elif surf_id == "sphere.reg":
        fs_surf_id, surf_id, surf_type, surf_secondary_type = (
            "sphere.reg",
            "sphere",
            "SPHERICAL",
            None,
        )
    else:
        raise ValueError(
            f"Unrecognized surface name: {surf_id} ...\n"
            "Valid option: 'wm', 'pial', 'sphere', 'sphere.reg'"
        )

    # Convert surface from FreeSurfer format to GIFTI format
    surf_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "surf", f"{hemi.lower()}h.{fs_surf_id}")
    fname = f"sub-{sub_id}_hemi-{hemi}_space-fsnative_den-fsnative_{surf_id}.surf.gii"
    if fs_surf_id == "sphere.reg":
        fname = fname.replace(f"space-fsnative", f"space-fsaverage")
    out_file = Path(out_dir).joinpath(fname)
    print(f"Converting {surf_file} ...", flush=True)

    # Set GIFTI metadata
    run_cmd(f"mris_convert {surf_file} {out_file}")
    set_structure_cmd = (
        f"wb_command -disable-provenance -set-structure {out_file} "
        f"{structure} -surface-type {surf_type}"
    )
    if surf_secondary_type:
        set_structure_cmd += f" -surface-secondary-type {surf_secondary_type}"
    run_cmd(set_structure_cmd)

    # Adjust CRAS if the matrix is supplied
    # A note from niworkflow.interfaces.surf.NormalizeSurf:
    # FreeSurfer includes an offset to the center of the brain
    # volume that is not respected by all software packages.
    # Normalization involves adding this offset to the coordinates
    # of all vertices, and zeroing out that offset, to ensure
    # consistent behavior across software packages.
    # In particular, this normalization is consistent with the Human
    # Connectome Project pipeline (see `AlgorithmSurfaceApplyAffine`
    # _ and `FreeSurfer2CaretConvertAndRegisterNonlinear`_),
    # although the the HCP may not zero out the offset.
    if adjust_cras:
        with tempfile.TemporaryDirectory() as tmp_dir:
            ref_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "mri", "brain.finalsurfs.mgz")
            cras_file = get_cras(ref_file, tmp_dir)
            print(f"Adjusting CRAS offset: {fname} ...", flush=True)
            run_cmd(
                "wb_command -disable-provenance -surface-apply-affine "
                f"{out_file} {cras_file} {out_file}"
            )
            # Cleanup CRAS codes in GIFTI metadata
            # See https://github.com/nipreps/niworkflows/blob/a2d3686bb9b184ec15e2147a3ae6f86c7e066929/niworkflows/interfaces/surf.py#L562
            # Using AFNI's gifti_tool
            reset_cras_cmd = f"gifti_tool -infile {out_file} -write_gifti {out_file}"
            for key in ["VolGeomC_R", "VolGeomC_A", "VolGeomC_S"]:
                reset_cras_cmd += f" -mod_DA_meta {key} 0.000000"
            run_cmd(reset_cras_cmd)
            # Output CRAS matrix if requested
            if debug:
                Path(out_dir).joinpath("temp_cras").mkdir(exist_ok=True)
                shutil.copy(
                    cras_file,
                    Path(out_dir).joinpath(
                        "temp_cras", f"sub-{sub_id}_hemi-{hemi}_{surf_id}_desc-cras_xfm.mat"
                    ),
                )

    # Apply affine transformation if the matrix is supplied
    if xfm_file:
        apply_affine_transformation_to_surface(out_file, xfm_file, out_dir, debug=debug)

    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file, gim_meta={"UserName": ""}, da_meta={"Name": ""})

    return out_file


def convert_freesurfer_metric(
    sub_id: str, hemi: Literal["L", "R"], metric_id: str, fs_dir: PathLike, out_dir: PathLike
) -> Path:
    """Converts FreeSurfer's metric map to GIFTI format.

    Args:
        sub_id: SubjectID.
        hemi: Brain hemisphere.
        metric_id: Surface metric name in FreeSurfer's outputs.
        fs_dir: Subject's FreeSurfer output directory.
        out_dir: Directory to store output file.

    Returns:
        A surface metric file in GIFTI format.

    Raises:
        ValueError: Unrecognized metric_id.
    """

    # Parse hemisphere
    hemi, structure = parse_hemi(hemi)

    # Parse metric
    if metric_id in ["sulc", "curv"]:
        palette_mode = "MODE_AUTO_SCALE_PERCENTAGE"
        palette_options = (
            "-pos-percent 2 98 -palette-name Gray_Interp -disp-pos true "
            "-disp-neg true -disp-zero true"
        )
    elif metric_id == "thickness":
        palette_mode = "MODE_AUTO_SCALE_PERCENTAGE"
        palette_options = (
            "-pos-percent 4 96 -interpolate true -palette-name videen_style "
            "-disp-pos true -disp-neg false -disp-zero false"
        )
    else:
        raise ValueError(f"Unrecognized metric: {metric_id} ...\n")

    # Convert metric file
    metric_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "surf", f"{hemi.lower()}h.{metric_id}")
    wm_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "surf", f"{hemi.lower()}h.white")
    out_file = Path(out_dir).joinpath(
        f"sub-{sub_id}_hemi-{hemi}_space-fsnative_den-fsnative_{metric_id}.shape.gii"
    )
    print(f"Converting metric: {metric_file} ...", flush=True)
    run_cmd(f"mris_convert -c {metric_file} {wm_file} {out_file}")

    # Set GIFTI metadata
    run_cmd("wb_command -disable-provenance -set-structure " f"{out_file} {structure}")
    run_cmd(
        f"wb_command -disable-provenance -metric-math 'var * -1' {out_file} -var var {out_file}"
    )
    run_cmd(
        f"wb_command -disable-provenance -set-map-names {out_file} "
        f"-map 1 sub-{sub_id}_hemi-{hemi}_{metric_id}"
    )
    run_cmd(
        "wb_command -disable-provenance -metric-palette "
        f"{out_file} {palette_mode} {palette_options}"
    )

    # Additional step for thickness metric
    # From https://github.com/Washington-University/HCPpipelines/blob/1334b35ab863540044333bbdec70a68fb19ab611/PostFreeSurfer/scripts/FreeSurfer2CaretConvertAndRegisterNonlinear.sh#L362
    if metric_id == "thickness":
        run_cmd(
            "wb_command -disable-provenance -metric-math 'abs(thickness)' "
            f"{out_file} -var thickness {out_file}"
        )

    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file, da_atr={"Intent": "NIFTI_INTENT_SHAPE"})

    return out_file


def convert_freesurfer_annot(
    sub_id: str, hemi: Literal["L", "R"], annot_id: str, fs_dir: PathLike, out_dir: PathLike
) -> Path:
    """Converts FreeSurfer's annotation data to GIFTI format.

    Args:
        sub_id: SubjectID.
        hemi: Brain hemisphere.
        annot_id: Surface annotation name in FreeSurfer's outputs.
        fs_dir: Subject's FreeSurfer output directory.
        out_dir: Directory to store output file.

    Returns:
        A surface label file in GIFTI format.

    Raises:
        ValueError: Unrecognized annot_id.
    """

    # Parse hemisphere
    hemi, structure = parse_hemi(hemi)

    # Parse annotation name
    if annot_id == "aparc":
        atlas_id = "Aparc"
    elif annot_id == "aparc.a2009s":
        atlas_id = "Destrieux"
    elif annot_id == "aparc.DKTatlas":
        atlas_id = "DKT"
    else:
        raise ValueError(f"Unrecognized annotation: {annot_id} ...\n")

    # Convert annotation file
    annot_file = Path(fs_dir).joinpath(
        f"sub-{sub_id}", "label", f"{hemi.lower()}h.{annot_id}.annot"
    )
    wm_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "surf", f"{hemi.lower()}h.white")
    out_file = Path(out_dir).joinpath(
        f"sub-{sub_id}_hemi-{hemi}_space-fsnative_den-fsnative_desc-{atlas_id}_dseg.label.gii"
    )
    print(f"Converting annotation: {annot_file} ...", flush=True)
    run_cmd(f"mris_convert --annot {annot_file} {wm_file} {out_file}")

    # Set GIFTI metadata
    run_cmd("wb_command -disable-provenance -set-structure " f"{out_file} {structure}")
    run_cmd(
        f"wb_command -disable-provenance -set-map-names {out_file} -map 1 "
        f"sub-{sub_id}_hemi-{hemi}_desc-{atlas_id}"
    )
    run_cmd(
        f"wb_command -disable-provenance -gifti-label-add-prefix {out_file} {hemi}_ {out_file}"
    )

    # Cleanup metadata
    _ = sanitize_gii_metadata(out_file, out_file)

    return out_file


def convert_freesurfer_volume(
    sub_id: str,
    volume_id: str,
    fs_dir: PathLike,
    out_dir: PathLike,
    xfm_file: Optional[PathLike] = None,
    ref_file: Optional[PathLike] = None,
    lut_file: Optional[PathLike] = None,
) -> Path:
    """Converts FreeSurfer's volume image to NIFTI format.

    Args:
        sub_id: SubjectID.
        volume_id: Volume image name in FreeSurfer's outputs.
        fs_dir: Subject's FreeSurfer output directory.
        out_dir: Directory to store output file.
        xfm_file: An ITK format affine transformation matrix file. If it
            is given, applying it to the volume file. Optional.
        ref_file: Reference volume file for xfm_file. Optional.
        lut_file: Lut file contains label information of FreeSurfer's
            parcellations. It is used to import label information to
            parcellation NIFTI image header. Optional.

    Returns:
        A volume image file.

    Raises:
        ValueError: ref_file is not specified when xfm_file is given.
    """

    # Conform name between FS and output
    names = {
        "T1": "T1w",
        "aparc+aseg": "AparcAseg",
        "aparc.a2009s+aseg": "DestrieuxAseg",
        "aparc.DKTatlas+aseg": "DKTAseg",
        "wmparc": "WMParc",
    }
    atlas_list = ["wmparc", "aparc.a2009s+aseg", "aparc+aseg", "aparc.DKTatlas+aseg"]

    # Convert volume from mgz to nifti
    # regular volume
    if not volume_id in atlas_list:
        in_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "mri", f"{volume_id}.mgz")
        out_file = Path(out_dir).joinpath(
            f"sub-{sub_id}_space-T1w_desc-FS_{names[volume_id]}.nii.gz"
        )
        print(f"Converting {in_file} ...", flush=True)
        run_cmd(f"mri_convert {in_file} {out_file}")
    # parcellation
    else:
        in_file = Path(fs_dir).joinpath(f"sub-{sub_id}", "mri", f"{volume_id}.mgz")
        out_file = Path(out_dir).joinpath(
            f"sub-{sub_id}_space-T1w_desc-{names[volume_id]}_dseg.nii.gz"
        )
        print(f"\nConverting {in_file} ...", flush=True)
        run_cmd(f"mri_convert {in_file} {out_file}")

    # Apply affine transformation to align volume to reference
    if xfm_file is not None:
        if ref_file is None:
            raise ValueError("If xfm_file is provided, ref_file is also required.")
        print(f"\nApply affine transformation: {out_file} ...", flush=True)
        cmd = f"antsApplyTransforms -i {out_file} -r {ref_file} -o {out_file} -t {xfm_file} "
        # regular volume
        if not volume_id in atlas_list:
            cmd += "-u float -n LanczosWindowedSinc"
        # parcellation
        else:
            cmd += "-u int -n MultiLabel"
        run_cmd(cmd)

    # Import label information to NIFTI file
    if (volume_id in atlas_list) and (lut_file is not None):
        print(f"\nImporting label information: {out_file} ...", flush=True)
        run_cmd(
            "wb_command -disable-provenance -logging SEVERE -volume-label-import "
            f"{out_file} {lut_file} {out_file} -drop-unused-labels"
        )

    return out_file


def get_cras(ref_file: PathLike, out_dir: PathLike) -> Path:
    """Writes FreeSurfer's CRAS matrix to file.

    Args:
        ref_file: A reference image file to get the CRAS matrix.
        out_dir: Directory to store output file.

    Returns:
        A text file contains CRAS matrix.
    """

    cras_file = Path(out_dir).joinpath(f"cras_xfm.mat")
    # Get cras infomation
    cras = run_cmd(f"mri_info --cras {ref_file}", print_output=False)
    cras = cras.stdout.replace("\n", "").split(" ")
    # Write out the cras infomation like an affine matrix
    with open(cras_file, "w") as f:
        f.write(f"1 0 0 {cras[0]}\n")
        f.write(f"0 1 0 {cras[1]}\n")
        f.write(f"0 0 1 {cras[2]}\n")
        f.write(f"0 0 0 1\n")

    return cras_file


def apply_affine_transformation_to_surface(
    surf_file: PathLike, itk_file: PathLike, out_dir: PathLike, debug: bool = False
) -> Path:
    """Applies affine transformation to a surface in GIFTI format.

    Args:
        surf_file: A surface file in GIFTI format.
        itk_file: Affine transformation matrix (ITK format) file.
        out_dir: Directory to store output file.
        debug: If true, output intermediate files to out_dir.

    Returns:
        A surface file in GIFTI format.
    """

    # Temporarily modify itk affine matrix header
    # The command `wb_command -convert-affine` is used for converting
    # itk format affine matrix to a NIFTI world affine.
    # In `wb_command`, the accepted itk affine class is
    # `MatrixOffsetTransformBase_double_3_3`, which is a base class in
    # itk specification. And this class should never be passed to a
    # downstream software.
    # See https://github.com/Washington-University/workbench/blob/f31a4edc490c3b8afa2ecca1e97390a53719fb33/src/Files/AffineFile.cxx#L150
    # for codes in `wb_command` source file.
    # In order to mitigate this issue, here we modify the affine matrix
    # created by fMRIprep manually to satisfy the `wb_command`.
    # Since the class `AffineTransform_float_3_3` is a subclass of
    # `MatrixOffsetTransformBase_double_3_3`, this modification should
    # be pretty safe.
    with tempfile.TemporaryDirectory() as tmp_dir:

        itk_mod_file = Path(tmp_dir).joinpath(
            Path(Path(itk_file).name.replace("_xfm", "_desc-modified_xfm"))
        )
        _ = _modify_itk_class(itk_file, itk_mod_file)

        # Convert transformation matrix from itk format to NIFTI 'world' affine
        world_affine_file = Path(tmp_dir).joinpath(
            Path(Path(itk_file).name.replace("_xfm", "_desc-world_xfm"))
        )
        run_cmd(
            "wb_command -convert-affine " f"-from-itk {itk_mod_file} -to-world {world_affine_file}"
        )

        # Apply affine transformation to surface
        print(f"Applying affine transformation to {Path(surf_file).name} ...", flush=True)
        run_cmd(
            "wb_command -disable-provenance -surface-apply-affine "
            f"{surf_file} {world_affine_file} {surf_file}"
        )

        # Output affine matrix if requested
        if debug:
            shutil.copytree(tmp_dir, Path(out_dir).joinpath("temp_affine"), dirs_exist_ok=True)

    return Path(surf_file)


def _modify_itk_class(in_file: PathLike, out_file: PathLike) -> Path:
    """Modifies ITK affine matrix class information."""

    with open(in_file, "r") as f:
        data = f.read()
    data = data.replace("AffineTransform_float_3_3", "MatrixOffsetTransformBase_double_3_3")
    with open(out_file, "w") as f:
        f.write(data)
    return Path(out_file)
