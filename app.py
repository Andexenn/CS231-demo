import gradio as gr
import os
import shutil
import subprocess
import tempfile
import torch
import sys
import glob
import pandas as pd

# Setup Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "inference_model"))

# Python executable for the UIT2024_medicare_env (has torch, ASlide, openslide, h5py, cv2)
# This env is used to run PrePATH subprocesses.
PREPATH_PYTHON = "/home/nguyenvd_drone/anaconda3/envs/UIT2024_medicare_env/bin/python"

# Attempt to load inference functions
try:
    from abmil import load_model as load_abmil, predict as predict_abmil
    from cemil import load_models as load_cemil, predict as predict_cemil
    from dsmil import load_model as load_dsmil, predict as predict_dsmil
except ImportError as e:
    print(f"Warning: Could not import models. {e}")

# Define paths to checkpoints
CKPT_DIR = os.path.join(BASE_DIR, "ckp")
abmil_ckpt = os.path.join(CKPT_DIR, "abmil_best.pth")
cemil_instructor_ckpt = os.path.join(CKPT_DIR, "instructor_best.pth")
cemil_learner_ckpt = os.path.join(CKPT_DIR, "learner_best.pth")
dsmil_ckpt = os.path.join(CKPT_DIR, "dsmil_best.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Pre-load models globally
print(f"Loading models to {device}...")
abmil_model = None
cemil_instructor, cemil_learner = None, None
dsmil_model = None

try:
    abmil_model = load_abmil(abmil_ckpt, device=device)
    cemil_instructor, cemil_learner = load_cemil(
        cemil_instructor_ckpt, cemil_learner_ckpt, device=device
    )
    dsmil_model = load_dsmil(dsmil_ckpt, device=device)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")


def format_probs(probs):
    """Format numpy probabilities nicely."""
    if isinstance(probs, list) or hasattr(probs, 'tolist'):
        probs = probs.tolist()
    return f"LUAD: {probs[0]:.4f}, LUSC: {probs[1]:.4f}"


def normalize_upload(wsi_files):
    """
    Normalize the Gradio file upload value to a flat list of path strings.

    Gradio 4+  with file_count="multiple" passes:
      - a list of str  (absolute temp-file paths) in Gradio >= 5
      - a list of file-like objects with a .name attribute in older versions
      - a single str / file-like when only one file is uploaded
      - None / [] when nothing is uploaded
    """
    if not wsi_files:
        return []

    # Gradio 5+ returns a list[str] directly; older returns list[NamedString]
    if isinstance(wsi_files, (str, bytes)):
        wsi_files = [wsi_files]

    paths = []
    for f in wsi_files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):           # NamedString / UploadedFile
            paths.append(f.name)
        elif hasattr(f, "orig_name"):      # some Gradio versions
            paths.append(f.orig_name)
        else:
            raise ValueError(f"Unknown file object type: {type(f)}")

    return [p for p in paths if p]        # drop any empty strings


def process_wsi(wsi_files):
    """
    Process one or more uploaded .svs files through the full pipeline:
      1. Segmentation & patching (PrePATH) for all slides in this batch.
      2. Feature extraction with ResNet50 for all slides in this batch.
      3. Per-slide inference with ABMIL, CEMIL, DSMIL.

    The CSV passed to the extractor is built from the exact set of slide IDs
    copied into wsi_dir, guaranteeing CSV ↔ wsi_dir ↔ patch_dir consistency.

    Args:
        wsi_files: value from gr.File(file_count="multiple")

    Returns:
        pd.DataFrame with columns [Slide, ABMIL, CEMIL, DSMIL]
    """
    empty_df = pd.DataFrame(columns=["Slide", "ABMIL", "CEMIL", "DSMIL"])

    try:
        file_paths = normalize_upload(wsi_files)
    except ValueError as e:
        return pd.DataFrame({"Error": [str(e)]})

    if not file_paths:
        return empty_df

    extract_dir = os.path.join(BASE_DIR, "extract")
    prepath_dir = os.path.join(extract_dir, "PrePATH")
    if not os.path.exists(prepath_dir):
        return pd.DataFrame({"Error": ["PrePATH directory not found."]})

    # -----------------------------------------------------------------------
    # Use a unique temp dir per request to allow concurrent runs without clash
    # -----------------------------------------------------------------------
    temp_dir = tempfile.mkdtemp(prefix="wsi_proc_", dir=BASE_DIR)

    wsi_dir  = os.path.join(temp_dir, "wsi")    # ephemeral: only this batch's WSIs
    csv_dir  = os.path.join(temp_dir, "csv")    # ephemeral: CSV for this batch
    feat_dir = os.path.join(temp_dir, "feats")  # ephemeral: extracted features
    os.makedirs(wsi_dir,  exist_ok=True)
    os.makedirs(csv_dir,  exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    # Persistent patch storage (masks, stitches, h5 coordinates).
    # Shared across runs so patching can be skipped for previously-seen slides.
    patch_dir = os.path.join(extract_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Copy all uploaded files to wsi_dir and derive slide IDs from the
    # actual filenames in wsi_dir — this is the single source of truth.
    # -----------------------------------------------------------------------
    slide_ids = []
    for src_path in file_paths:
        fname = os.path.basename(src_path)
        # Ensure .svs extension
        if not fname.lower().endswith(".svs"):
            fname = os.path.splitext(fname)[0] + ".svs"
        dst_path = os.path.join(wsi_dir, fname)
        shutil.copy2(src_path, dst_path)
        slide_id = os.path.splitext(fname)[0]
        slide_ids.append(slide_id)

    # Deduplicate while preserving order
    seen = set()
    unique_slide_ids = []
    for sid in slide_ids:
        if sid not in seen:
            seen.add(sid)
            unique_slide_ids.append(sid)
    slide_ids = unique_slide_ids

    print(f"\n{'='*60}")
    print(f"Processing {len(slide_ids)} slide(s): {slide_ids}")
    print(f"wsi_dir  : {wsi_dir}")
    print(f"patch_dir: {patch_dir}")
    print(f"feat_dir : {feat_dir}")
    print(f"{'='*60}\n")

    results = []

    try:
        # ------------------------------------------------------------------
        # Step 1 — Segmentation & Patching
        # Scans wsi_dir; writes .h5 patch coords into patch_dir/patches/
        # ------------------------------------------------------------------
        print("Running PrePATH patching...")
        cmd_patch = [
            PREPATH_PYTHON, "create_patches_fp.py",
            "--source",      wsi_dir,
            "--save_dir",    patch_dir,
            "--patch_size",  "256",
            "--preset",      "tcga.csv",
            "--patch_level", "0",
            "--wsi_format",  "svs",
            "--seg", "--patch", "--stitch"
        ]
        subprocess.run(cmd_patch, cwd=prepath_dir, check=True)

        # ------------------------------------------------------------------
        # Step 2 — Build CSV that exactly mirrors the wsi_dir contents.
        #
        # We enumerate the actual .svs files written to wsi_dir (not the
        # original upload list) so that the CSV, wsi_dir, and the patch .h5
        # files are guaranteed to be consistent.
        # ------------------------------------------------------------------
        print("Generating CSV...")
        actual_svs = sorted(glob.glob(os.path.join(wsi_dir, "*.svs")))
        if not actual_svs:
            return pd.DataFrame({"Error": ["No .svs files found in wsi_dir after copy."]})

        # Rebuild slide_ids from what is actually on disk (canonical)
        slide_ids = [os.path.splitext(os.path.basename(p))[0] for p in actual_svs]

        # Only include slides whose .h5 patch file was produced by PrePATH
        valid_slide_ids = []
        for sid in slide_ids:
            h5_path = os.path.join(patch_dir, "patches", f"{sid}.h5")
            if os.path.exists(h5_path):
                valid_slide_ids.append(sid)
            else:
                print(f"  [WARN] No .h5 patch file for '{sid}' — skipping in CSV.")
                results.append({
                    "Slide": sid,
                    "ABMIL":  "Patching failed / no patches",
                    "CEMIL":  "Patching failed / no patches",
                    "DSMIL":  "Patching failed / no patches",
                })

        if not valid_slide_ids:
            return pd.DataFrame({"Error": ["Patching produced no valid .h5 files."]})

        csv_path = os.path.join(csv_dir, "part_0.csv")
        with open(csv_path, "w") as f:
            f.write("case_id,slide_id\n")
            for sid in valid_slide_ids:
                f.write(f'"{sid}","{sid}"\n')

        print(f"  CSV written with {len(valid_slide_ids)} slide(s): {valid_slide_ids}")

        # ------------------------------------------------------------------
        # Step 3 — Feature Extraction
        # Uses csv_path (only valid slides) and data_slide_dir=wsi_dir
        # so slide_id → WSI path lookup cannot KeyError on stale entries.
        # ------------------------------------------------------------------
        print("Extracting features with ResNet50...")
        cmd_feat = [
            PREPATH_PYTHON, "extract_features_fp_fast.py",
            "--model",          "resnet50",
            "--csv_path",       csv_path,
            "--data_coors_dir", patch_dir,
            "--data_slide_dir", wsi_dir,
            "--feat_dir",       feat_dir,
            "--ignore_partial", "yes",
            "--batch_size",     "128",
            "--datatype",       "auto",
            "--slide_ext",      ".svs",
            "--save_storage",   "yes"
        ]
        subprocess.run(cmd_feat, cwd=prepath_dir, check=True)

        # ------------------------------------------------------------------
        # Step 4 — Per-slide inference
        # ------------------------------------------------------------------
        for slide_id in valid_slide_ids:
            pt_path = os.path.join(feat_dir, "pt_files", "resnet50", f"{slide_id}.pt")

            if not os.path.exists(pt_path):
                print(f"  [WARN] No .pt file found for '{slide_id}', skipping inference.")
                results.append({
                    "Slide": slide_id,
                    "ABMIL": "Feature extraction failed",
                    "CEMIL": "Feature extraction failed",
                    "DSMIL": "Feature extraction failed",
                })
                continue

            features = torch.load(pt_path, map_location=device)
            if not isinstance(features, torch.Tensor):
                features = torch.tensor(features)
            features = features.float()
            if features.dim() > 2:
                features = features.squeeze(0)

            print(f"  [{slide_id}] Features shape: {features.shape}")

            res_abmil = "Model not loaded"
            if abmil_model:
                _, label, probs, _ = predict_abmil(abmil_model, features, device=device)
                res_abmil = f"{label} ({format_probs(probs)})"

            res_cemil = "Model not loaded"
            if cemil_instructor and cemil_learner:
                _, label, probs, _, _, _ = predict_cemil(
                    cemil_instructor, cemil_learner, features, device=device
                )
                res_cemil = f"{label} ({format_probs(probs)})"

            res_dsmil = "Model not loaded"
            if dsmil_model:
                _, label, probs, _ = predict_dsmil(dsmil_model, features, device=device)
                res_dsmil = f"{label} ({format_probs(probs)})"

            results.append({
                "Slide": slide_id,
                "ABMIL": res_abmil,
                "CEMIL": res_cemil,
                "DSMIL": res_dsmil,
            })

    except subprocess.CalledProcessError as e:
        return pd.DataFrame({"Error": [f"Subprocess failed: {e}"]})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return pd.DataFrame({"Error": [f"Error: {str(e)}"]})
    finally:
        # Clean up the per-request ephemeral workspace.
        # Persistent patches in extract/patches are left intact.
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temp dir: {temp_dir}")

    return pd.DataFrame(results) if results else empty_df


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
# Using light theme as requested
with gr.Blocks(title="NSCLC WSI Classifier", theme=gr.themes.Default()) as demo:
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("# Demo: Multiple Instance Learning for NSCLC Classification")
            gr.Markdown("### Course: CS231 | Major: Software Engineering")
        with gr.Column(scale=2):
            # Optional: Add a small placeholder or logo if needed, otherwise just spacing
            pass

    gr.Markdown(
        "This application demonstrates the end-to-end processing and classification of "
        "Non-Small Cell Lung Cancer (NSCLC) Whole Slide Images. Supported types: **LUAD** and **LUSC**.\n\n"
        "**Pipeline Workflow:**\n"
        "1. **PrePATH:** Tissue segmentation, patching (256px), and stitching.\n"
        "2. **ResNet50:** Visual feature extraction from each tissue patch.\n"
        "3. **MIL Models:** Inference using **ABMIL**, **CEMIL**, and **DSMIL** architectures."
    )

    with gr.Row():
        wsi_input = gr.File(
            label="Upload .svs WSI(s) — Multiple files supported",
            file_types=[".svs"],
            file_count="multiple",
        )

    run_btn = gr.Button("🚀 Extract Features & Predict", variant="primary")

    results_table = gr.Dataframe(
        headers=["Slide", "ABMIL", "CEMIL", "DSMIL"],
        label="Diagnostic Predictions (LUAD vs LUSC)",
        wrap=True,
        interactive=False,
    )

    run_btn.click(
        fn=process_wsi,
        inputs=[wsi_input],
        outputs=[results_table],
    )

if __name__ == "__main__":
    # Launch on all interfaces for remote access
    demo.launch(server_name="0.0.0.0", server_port=7890)
