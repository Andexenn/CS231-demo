import gradio as gr
import os
import json
import shutil
import subprocess
import tempfile
import torch
import sys
import glob
import pandas as pd
import numpy as np
from datetime import datetime

# ---------------------------------------------------------------------------
# Setup Base Directories
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "inference_model"))

LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

# Python executable for the UIT2024_medicare_env (has torch, ASlide, openslide, h5py, cv2)
PREPATH_PYTHON = "/home/nguyenvd_drone/anaconda3/envs/UIT2024_medicare_env/bin/python"

# ---------------------------------------------------------------------------
# Load split manifest for ground-truth labels
# ---------------------------------------------------------------------------
MANIFEST_PATH = os.path.join(BASE_DIR, "split_manifest.csv")

# Regex to strip the luad__ / lusc__ prefix from manifest filenames
import re
_PREFIX_RE = re.compile(r'^(luad|lusc)__', re.IGNORECASE)

def load_label_map(manifest_path):
    """
    Build TWO lookup dicts from split_manifest.csv:
      - LABEL_MAP_FULL : full stem (with prefix, without .pt extension) -> label
          e.g. 'luad__TCGA-05-4244-01Z-00-DX1.d4ff32cd' -> 'LUAD'
      - LABEL_MAP_BARE : bare TCGA stem (prefix stripped, no .pt extension) -> label
          e.g. 'TCGA-05-4244-01Z-00-DX1.d4ff32cd' -> 'LUAD'
    This covers inputs that do or do not carry the luad__/lusc__ prefix.
    """
    full_map = {}   # keyed by 'luad__TCGA-xxx'
    bare_map = {}   # keyed by 'TCGA-xxx'
    if not os.path.exists(manifest_path):
        print(f"[WARN] Manifest not found at {manifest_path}")
        return full_map, bare_map
    df = pd.read_csv(manifest_path)
    for _, row in df.iterrows():
        # strip the .pt extension
        stem = os.path.splitext(str(row["filename"]))[0]
        label = str(row["label"]).upper()  # 'LUAD' or 'LUSC'
        full_map[stem] = label
        # also index without the luad__/lusc__ prefix
        bare_stem = _PREFIX_RE.sub("", stem)
        bare_map[bare_stem] = label
    return full_map, bare_map

LABEL_MAP_FULL, LABEL_MAP_BARE = load_label_map(MANIFEST_PATH)
print(f"Loaded label map: {len(LABEL_MAP_FULL)} entries (full), {len(LABEL_MAP_BARE)} entries (bare).")

def _strip_known_ext(name):
    """
    Strip only known WSI / feature-file extensions from a filename.
    NEVER uses os.path.splitext blindly because TCGA slide IDs contain
    dots followed by UUIDs (e.g. '.2f0b6cea-795a-40ad-93a9-319858e6fb3b')
    which os.path.splitext would wrongly treat as an extension.
    """
    _KNOWN_EXTS = ('.svs', '.pt', '.ndpi', '.tiff', '.tif',
                   '.scn', '.bif', '.mrxs', '.svslide', '.vms', '.vmu')
    for ext in _KNOWN_EXTS:
        if name.lower().endswith(ext):
            return name[: -len(ext)]
    return name


def get_ground_truth(slide_id):
    """
    Look up the ground-truth label for a slide_id.

    The manifest stores filenames like:
        luad__TCGA-44-7662-01Z-00-DX1.2f0b6cea-795a-40ad-93a9-319858e6fb3b.pt
    The 'bare' key (in LABEL_MAP_BARE) is:
        TCGA-44-7662-01Z-00-DX1.2f0b6cea-795a-40ad-93a9-319858e6fb3b

    Input slide_id can be:
      - Bare TCGA ID:                 TCGA-44-7662-01Z-00-DX1.2f0b6cea-...
      - With prefix:                  luad__TCGA-44-7662-01Z-00-DX1.2f0b6cea-...
      - With or without extension:    ...svs / ...pt / nothing

    Strategy (all O(1) dict lookups):
      1. Strip known file extension (.svs / .pt / etc.) if present
      2. Try LABEL_MAP_FULL  (with prefix)
      3. Strip luad__/lusc__ prefix, try LABEL_MAP_BARE
      4. Case-insensitive fallback on LABEL_MAP_BARE
    """
    # Step 1: strip extension ONLY if it's a known file type
    slide_id = _strip_known_ext(slide_id)

    # Step 2: has prefix → full map
    if slide_id in LABEL_MAP_FULL:
        return LABEL_MAP_FULL[slide_id]

    # Step 3: strip prefix → bare map
    bare_id = _PREFIX_RE.sub("", slide_id)
    if bare_id in LABEL_MAP_BARE:
        return LABEL_MAP_BARE[bare_id]

    # Step 4: case-insensitive fallback
    bare_id_lower = bare_id.lower()
    for key, label in LABEL_MAP_BARE.items():
        if key.lower() == bare_id_lower:
            return label

    return "Unknown"

# ---------------------------------------------------------------------------
# Load inference models
# ---------------------------------------------------------------------------
try:
    from abmil import load_model as load_abmil, predict as predict_abmil
    from cemil import load_models as load_cemil, predict as predict_cemil
    from dsmil import load_model as load_dsmil, predict as predict_dsmil
except ImportError as e:
    print(f"Warning: Could not import models. {e}")

CKPT_DIR = os.path.join(BASE_DIR, "ckp")
abmil_ckpt = os.path.join(CKPT_DIR, "abmil_best.pth")
cemil_instructor_ckpt = os.path.join(CKPT_DIR, "instructor_best.pth")
cemil_learner_ckpt = os.path.join(CKPT_DIR, "learner_best.pth")
dsmil_ckpt = os.path.join(CKPT_DIR, "dsmil_best.pth")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------
def format_probs(probs):
    """Format numpy probabilities nicely."""
    if isinstance(probs, list) or hasattr(probs, 'tolist'):
        probs = probs.tolist() if not isinstance(probs, list) else probs
    return f"LUAD: {probs[0]:.4f}, LUSC: {probs[1]:.4f}"


def normalize_upload(wsi_files):
    """
    Normalize the Gradio file upload value to a flat list of path strings.
    """
    if not wsi_files:
        return []
    if isinstance(wsi_files, (str, bytes)):
        wsi_files = [wsi_files]
    paths = []
    for f in wsi_files:
        if isinstance(f, str):
            paths.append(f)
        elif hasattr(f, "name"):
            paths.append(f.name)
        elif hasattr(f, "orig_name"):
            paths.append(f.orig_name)
        else:
            raise ValueError(f"Unknown file object type: {type(f)}")
    return [p for p in paths if p]


def save_log(slide_id, model_name, log_data):
    """
    Save a JSON diagnostic log for a given slide and model.
    Path: logs/<slide_id>/<model_name>.json
    """
    slide_log_dir = os.path.join(LOGS_DIR, slide_id)
    os.makedirs(slide_log_dir, exist_ok=True)
    log_path = os.path.join(slide_log_dir, f"{model_name}.json")

    # Convert numpy arrays to lists for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        return obj

    serializable = {}
    for k, v in log_data.items():
        if isinstance(v, np.ndarray):
            serializable[k] = v.tolist()
        elif isinstance(v, (np.float32, np.float64)):
            serializable[k] = float(v)
        elif isinstance(v, list):
            serializable[k] = [to_serializable(x) for x in v]
        else:
            serializable[k] = v

    with open(log_path, "w") as f:
        json.dump(serializable, f, indent=2)
    return log_path


# ---------------------------------------------------------------------------
# Core processing function
# ---------------------------------------------------------------------------
def process_wsi(wsi_files, local_paths_text):
    """
    Process slides from either Gradio upload or local disk paths.
    Returns a DataFrame with [Slide, Ground Truth, ABMIL, CEMIL, DSMIL].
    """
    empty_df = pd.DataFrame(columns=["Slide", "Ground Truth", "ABMIL", "CEMIL", "DSMIL"])

    # --- Collect file paths from both sources ---
    file_paths = []

    # Source 1: Gradio upload
    try:
        uploaded = normalize_upload(wsi_files)
        file_paths.extend(uploaded)
    except ValueError as e:
        return pd.DataFrame({"Error": [str(e)]})

    # Source 2: Local disk paths (one per line)
    if local_paths_text and local_paths_text.strip():
        for line in local_paths_text.strip().splitlines():
            line = line.strip()
            if line and os.path.exists(line):
                file_paths.append(line)
            elif line:
                print(f"[WARN] Local path does not exist, skipping: {line}")

    if not file_paths:
        return empty_df

    extract_dir = os.path.join(BASE_DIR, "extract")
    prepath_dir = os.path.join(extract_dir, "PrePATH")
    if not os.path.exists(prepath_dir):
        return pd.DataFrame({"Error": ["PrePATH directory not found."]})

    temp_dir = tempfile.mkdtemp(prefix="wsi_proc_", dir=BASE_DIR)
    wsi_dir  = os.path.join(temp_dir, "wsi")
    csv_dir  = os.path.join(temp_dir, "csv")
    feat_dir = os.path.join(temp_dir, "feats")
    os.makedirs(wsi_dir,  exist_ok=True)
    os.makedirs(csv_dir,  exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    patch_dir = os.path.join(extract_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    slide_ids = []
    for src_path in file_paths:
        fname = os.path.basename(src_path)
        if not fname.lower().endswith(".svs"):
            fname = os.path.splitext(fname)[0] + ".svs"
        dst_path = os.path.join(wsi_dir, fname)
        # Use a symlink instead of copying — avoids double-copying huge WSI files
        # through Gradio's temp directory which can corrupt or truncate the file.
        src_abs = os.path.abspath(src_path)
        if os.path.exists(dst_path) or os.path.islink(dst_path):
            os.remove(dst_path)
        os.symlink(src_abs, dst_path)
        print(f"  Linked: {src_abs} -> {dst_path}")
        slide_id = os.path.splitext(fname)[0]
        slide_ids.append(slide_id)

    # Deduplicate
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
        # Step 2 — Build CSV
        # ------------------------------------------------------------------
        print("Generating CSV...")
        actual_svs = sorted(glob.glob(os.path.join(wsi_dir, "*.svs")))
        if not actual_svs:
            return pd.DataFrame({"Error": ["No .svs files found in wsi_dir after copy."]})

        slide_ids = [os.path.splitext(os.path.basename(p))[0] for p in actual_svs]

        valid_slide_ids = []
        for sid in slide_ids:
            h5_path = os.path.join(patch_dir, "patches", f"{sid}.h5")
            if os.path.exists(h5_path):
                valid_slide_ids.append(sid)
            else:
                print(f"  [WARN] No .h5 patch file for '{sid}' — skipping in CSV.")
                results.append({
                    "Slide": sid,
                    "Ground Truth": get_ground_truth(sid),
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
        # Step 4 — Per-slide inference + logging
        # ------------------------------------------------------------------
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for slide_id in valid_slide_ids:
            pt_path = os.path.join(feat_dir, "pt_files", "resnet50", f"{slide_id}.pt")
            ground_truth = get_ground_truth(slide_id)

            if not os.path.exists(pt_path):
                print(f"  [WARN] No .pt file found for '{slide_id}', skipping inference.")
                results.append({
                    "Slide": slide_id,
                    "Ground Truth": ground_truth,
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

            # --- ABMIL ---
            res_abmil = "Model not loaded"
            if abmil_model:
                _, label, probs, attention = predict_abmil(abmil_model, features, device=device)
                res_abmil = f"{label} ({format_probs(probs)})"

                log_data = {
                    "timestamp": timestamp,
                    "slide_id": slide_id,
                    "ground_truth": ground_truth,
                    "model": "ABMIL",
                    "predicted_label": label,
                    "probabilities": {"LUAD": float(probs[0]), "LUSC": float(probs[1])},
                    "attention_weights": attention,
                    "attention_stats": {
                        "min": float(attention.min()),
                        "max": float(attention.max()),
                        "mean": float(attention.mean()),
                        "std": float(attention.std()),
                        "num_patches": int(attention.shape[0]),
                        "top5_patch_indices": attention.argsort()[::-1][:5].tolist(),
                    },
                }
                save_log(slide_id, "abmil", log_data)

            # --- CEMIL ---
            res_cemil = "Model not loaded"
            if cemil_instructor and cemil_learner:
                _, label, probs, top_indices, A_instructor, A_learner = predict_cemil(
                    cemil_instructor, cemil_learner, features, device=device
                )
                res_cemil = f"{label} ({format_probs(probs)})"

                log_data = {
                    "timestamp": timestamp,
                    "slide_id": slide_id,
                    "ground_truth": ground_truth,
                    "model": "CEMIL",
                    "predicted_label": label,
                    "probabilities": {"LUAD": float(probs[0]), "LUSC": float(probs[1])},
                    "top_k_patch_indices": top_indices,
                    "instructor_attention": A_instructor,
                    "instructor_attention_stats": {
                        "min": float(A_instructor.min()),
                        "max": float(A_instructor.max()),
                        "mean": float(A_instructor.mean()),
                        "num_patches": int(A_instructor.shape[0]),
                    },
                    "learner_attention": A_learner,
                    "learner_attention_stats": {
                        "min": float(A_learner.min()),
                        "max": float(A_learner.max()),
                        "mean": float(A_learner.mean()),
                        "num_patches": int(A_learner.shape[0]),
                    },
                }
                save_log(slide_id, "cemil", log_data)

            # --- DSMIL ---
            res_dsmil = "Model not loaded"
            if dsmil_model:
                with torch.no_grad():
                    feats = features.squeeze(0).to(device)
                    ins_scores, bag_prediction, A, B = dsmil_model(feats)
                    max_prediction, _ = torch.max(ins_scores, 0)
                    score = 0.5 * torch.sigmoid(bag_prediction) + 0.5 * torch.sigmoid(max_prediction.view(1, -1))
                    probs_tensor = torch.softmax(score, dim=1).squeeze()
                    probs = probs_tensor.cpu().numpy()
                    pred = int(np.argmax(probs))
                    label = {0: "LUAD", 1: "LUSC"}[pred]

                    # Detailed raw scores for logging
                    bag_pred_sigmoid  = torch.sigmoid(bag_prediction).squeeze().cpu().numpy()
                    max_pred_sigmoid  = torch.sigmoid(max_prediction).squeeze().cpu().numpy()
                    ins_scores_np     = ins_scores.cpu().numpy()
                    attention_np      = A.squeeze().cpu().numpy()

                res_dsmil = f"{label} ({format_probs(probs)})"

                log_data = {
                    "timestamp": timestamp,
                    "slide_id": slide_id,
                    "ground_truth": ground_truth,
                    "model": "DSMIL",
                    "predicted_label": label,
                    "probabilities": {"LUAD": float(probs[0]), "LUSC": float(probs[1])},
                    # i_classifier (instance-level scores per patch, shape [num_patches, 2])
                    "i_classifier_scores_raw": ins_scores_np,
                    "i_classifier_max_instance_sigmoid": {
                        "LUAD": float(max_pred_sigmoid[0]) if max_pred_sigmoid.ndim > 0 else float(max_pred_sigmoid),
                        "LUSC": float(max_pred_sigmoid[1]) if (max_pred_sigmoid.ndim > 0 and len(max_pred_sigmoid) > 1) else None,
                    },
                    # b_classifier (bag-level scores, shape [1, 2])
                    "b_classifier_bag_prediction_sigmoid": {
                        "LUAD": float(bag_pred_sigmoid[0]) if bag_pred_sigmoid.ndim > 0 else float(bag_pred_sigmoid),
                        "LUSC": float(bag_pred_sigmoid[1]) if (bag_pred_sigmoid.ndim > 0 and len(bag_pred_sigmoid) > 1) else None,
                    },
                    # Combined score (0.5 * bag + 0.5 * max-instance before softmax)
                    "combined_score_before_softmax": score.squeeze().cpu().numpy(),
                    # Attention weights from b_classifier
                    "b_classifier_attention": attention_np,
                    "b_classifier_attention_stats": {
                        "min": float(attention_np.min()),
                        "max": float(attention_np.max()),
                        "mean": float(attention_np.mean()),
                        "num_patches": int(attention_np.shape[0]) if attention_np.ndim > 0 else 1,
                        "top5_patch_indices": attention_np.argsort()[::-1][:5].tolist() if attention_np.ndim > 0 else [],
                    },
                }
                save_log(slide_id, "dsmil", log_data)

            results.append({
                "Slide": slide_id,
                "Ground Truth": ground_truth,
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
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Cleaned up temp dir: {temp_dir}")

    return pd.DataFrame(results) if results else empty_df


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title="NSCLC WSI Classifier", theme=gr.themes.Default()) as demo:
    with gr.Row():
        with gr.Column(scale=8):
            gr.Markdown("# Demo: Multiple Instance Learning for NSCLC Classification")
            gr.Markdown("### Course: CS231 | Major: Software Engineering")

    gr.Markdown(
        "This application demonstrates the end-to-end processing and classification of "
        "Non-Small Cell Lung Cancer (NSCLC) Whole Slide Images. Supported types: **LUAD** and **LUSC**.\n\n"
        "**Pipeline Workflow:**\n"
        "1. **PrePATH:** Tissue segmentation, patching (256px), and stitching.\n"
        "2. **ResNet50:** Visual feature extraction from each tissue patch.\n"
        "3. **MIL Models:** Inference using **ABMIL**, **CEMIL**, and **DSMIL** architectures.\n\n"
        "**Diagnostic logs** (scores, attention weights, etc.) are saved to the `logs/` folder after each run."
    )

    gr.Markdown("---")
    gr.Markdown("### Step 1: Select Input Method")

    with gr.Tabs():
        with gr.Tab("🌐 Upload via Browser"):
            wsi_input = gr.File(
                label="Upload .svs WSI file(s) — Multiple files supported",
                file_types=[".svs"],
                file_count="multiple",
            )
            gr.Markdown(
                "> ⚠️ **Note:** Uploading large WSI files over the internet can be slow. "
                "Use the **Local Disk** tab if the files are already on the server."
            )

        with gr.Tab("💾 Load from Local Disk (Faster)"):
            local_paths_input = gr.Textbox(
                label="Enter absolute paths to .svs files (one per line)",
                placeholder="/path/to/slide1.svs\n/path/to/slide2.svs",
                lines=5,
            )
            gr.Markdown(
                "> ✅ **Recommended** when running on the same server as the data. "
                "No upload required — files are read directly from disk."
            )

    gr.Markdown("---")
    run_btn = gr.Button("🚀 Extract Features & Predict", variant="primary")

    results_table = gr.Dataframe(
        headers=["Slide", "Ground Truth", "ABMIL", "CEMIL", "DSMIL"],
        label="Diagnostic Predictions (LUAD vs LUSC)",
        wrap=True,
        interactive=False,
    )

    run_btn.click(
        fn=process_wsi,
        inputs=[wsi_input, local_paths_input],
        outputs=[results_table],
    )

if __name__ == "__main__":
    # Launch on all interfaces for remote access
    demo.launch(server_name="0.0.0.0", server_port=7899, share=True)
