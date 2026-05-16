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
import re
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
_PREFIX_RE = re.compile(r'^(luad|lusc)__', re.IGNORECASE)

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
        stem = _strip_known_ext(str(row["filename"]))
        label = str(row["label"]).upper()  # 'LUAD' or 'LUSC'
        full_map[stem] = label
        # also index without the luad__/lusc__ prefix
        bare_stem = _PREFIX_RE.sub("", stem)
        bare_map[bare_stem] = label
    return full_map, bare_map

LABEL_MAP_FULL, LABEL_MAP_BARE = load_label_map(MANIFEST_PATH)
print(f"Loaded label map: {len(LABEL_MAP_FULL)} entries (full), {len(LABEL_MAP_BARE)} entries (bare).")

def get_ground_truth(slide_id):
    """
    Look up the ground-truth label for a slide_id.
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
    """
    slide_log_dir = os.path.join(LOGS_DIR, slide_id)
    os.makedirs(slide_log_dir, exist_ok=True)
    log_path = os.path.join(slide_log_dir, f"{model_name}.json")

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
    """
    empty_df = pd.DataFrame(columns=["Slide", "Ground Truth", "ABMIL", "CEMIL", "DSMIL"])
    file_paths = []

    # Source 1: Gradio upload
    try:
        uploaded = normalize_upload(wsi_files)
        file_paths.extend(uploaded)
    except ValueError as e:
        return pd.DataFrame({"Error": [str(e)]})

    # Source 2: Local disk paths
    if local_paths_text and local_paths_text.strip():
        for line in local_paths_text.strip().splitlines():
            line = line.strip()
            if not line: continue
            if os.path.exists(line):
                file_paths.append(line)
            else:
                potential_path = os.path.join(BASE_DIR, "images", line)
                if os.path.exists(potential_path):
                    file_paths.append(potential_path)
                elif not line.lower().endswith(".svs"):
                    potential_path_svs = potential_path + ".svs"
                    if os.path.exists(potential_path_svs):
                        file_paths.append(potential_path_svs)
                    else:
                        print(f"[WARN] Could not find file: {line}")
                else:
                    print(f"[WARN] Could not find file: {line}")

    if not file_paths:
        return empty_df

    extract_dir = os.path.join(BASE_DIR, "extract")
    prepath_dir = os.path.join(extract_dir, "PrePATH")
    if not os.path.exists(prepath_dir):
        return pd.DataFrame({"Error": ["PrePATH directory not found."]})

    # Use system /tmp instead of BASE_DIR (network mount) to ensure 
    # OpenSlide/ASlide can read the file without "Unsupported" errors.
    # Browser uploads work because they are stored in /tmp; we mimic that here.
    temp_dir = tempfile.mkdtemp(prefix="wsi_proc_")
    wsi_dir  = os.path.join(temp_dir, "wsi")
    csv_dir  = os.path.join(temp_dir, "csv")
    feat_dir = os.path.join(temp_dir, "feats")
    os.makedirs(wsi_dir,  exist_ok=True)
    os.makedirs(csv_dir,  exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    patch_dir = os.path.join(extract_dir, "patches")
    os.makedirs(patch_dir, exist_ok=True)

    slide_ids = []
    id_to_temp_name = {}
    
    for i, src_path in enumerate(file_paths):
        # 1. Original ID (preserves the long TCGA name)
        fname_orig = os.path.basename(src_path)
        if not fname_orig.lower().endswith(".svs"):
            fname_orig = os.path.splitext(fname_orig)[0] + ".svs"
            
        slide_id = _strip_known_ext(fname_orig)
        slide_ids.append(slide_id)
        
        # 2. Use the ORIGINAL name in /tmp (Browser uploads do this, so it should work)
        temp_name = fname_orig
        id_to_temp_name[slide_id] = slide_id 
        dst_path = os.path.join(wsi_dir, temp_name)
        
        # Physical Copy to /tmp
        src_abs = os.path.abspath(src_path)
        if os.path.exists(dst_path): os.remove(dst_path)
        
        try:
            shutil.copy2(src_abs, dst_path)
            # FORCE PERMISSIONS: Ensure the processing script can read it
            os.chmod(dst_path, 0o666)
            
            fsize = os.path.getsize(dst_path)
            print(f"  Copied to local /tmp: {dst_path} ({fsize/1024/1024:.2f} MB)")
            
            # Verify it's a valid TIFF/SVS (TIFF files start with 'II*' or 'MM*')
            with open(dst_path, 'rb') as f:
                header = f.read(4)
                print(f"  File header: {header}")
                if header not in [b'II\x2a\x00', b'MM\x00\x2a']:
                    print("  [WARN] File does not look like a standard TIFF/SVS!")
        except Exception as e:
            print(f"  [ERROR] Failed to copy {slide_id}: {e}")
            continue
            
        # Verify readability
        if not os.path.exists(dst_path) or os.path.getsize(dst_path) == 0:
            print(f"  [ERROR] File {dst_path} is missing or empty after copy!")
            continue
            
        for folder in ["patches", "masks", "stitches"]:
            for name_to_clean in [slide_id, f"slide_{i}"]:
                for ext in [".h5", ".png"]:
                    p = os.path.join(patch_dir, folder, f"{name_to_clean}{ext}")
                    if os.path.exists(p): os.remove(p)

    print(f"\nProcessing {len(slide_ids)} slide(s): {slide_ids}\n")

    results = []

    try:
        # Step 1 — Patching
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

        # Step 2 — Build CSV
        temp_to_orig = {v: k for k, v in id_to_temp_name.items()}
        valid_slide_ids = []
        for temp_sid in id_to_temp_name.values():
            orig_sid = temp_to_orig[temp_sid]
            h5_path = os.path.join(patch_dir, "patches", f"{temp_sid}.h5")
            if os.path.exists(h5_path):
                valid_slide_ids.append((temp_sid, orig_sid))
            else:
                print(f"  [WARN] No .h5 patch file for '{orig_sid}'")
                results.append({
                    "Slide": orig_sid, "Ground Truth": get_ground_truth(orig_sid),
                    "ABMIL": "Patching failed", "CEMIL": "Patching failed", "DSMIL": "Patching failed"
                })

        if not valid_slide_ids:
            return pd.DataFrame(results) if results else pd.DataFrame({"Error": ["No patches generated."]})

        csv_path = os.path.join(csv_dir, "part_0.csv")
        with open(csv_path, "w") as f:
            f.write("case_id,slide_id\n")
            for tsid, _ in valid_slide_ids:
                f.write(f'"{tsid}","{tsid}"\n')

        # Step 3 — Feature Extraction
        print("Extracting features...")
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

        # Step 4 — Inference
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for tsid, osid in valid_slide_ids:
            pt_path = os.path.join(feat_dir, "pt_files", "resnet50", f"{tsid}.pt")
            gt = get_ground_truth(osid)

            if not os.path.exists(pt_path):
                results.append({"Slide": osid, "Ground Truth": gt, "ABMIL": "Feat fail", "CEMIL": "Feat fail", "DSMIL": "Feat fail"})
                continue

            features = torch.load(pt_path, map_location=device).float()
            if features.dim() > 2: features = features.squeeze(0)

            # ABMIL
            res_abmil = "N/A"
            if abmil_model:
                _, label, probs, att = predict_abmil(abmil_model, features, device=device)
                res_abmil = f"{label} ({format_probs(probs)})"
                save_log(osid, "abmil", {"timestamp": timestamp, "slide_id": osid, "ground_truth": gt, "model": "ABMIL", "predicted_label": label, "probabilities": probs.tolist(), "attention_weights": att.tolist()})

            # CEMIL
            res_cemil = "N/A"
            if cemil_instructor and cemil_learner:
                _, label, probs, top_idx, A_i, A_l = predict_cemil(cemil_instructor, cemil_learner, features, device=device)
                res_cemil = f"{label} ({format_probs(probs)})"
                save_log(osid, "cemil", {"timestamp": timestamp, "slide_id": osid, "ground_truth": gt, "model": "CEMIL", "predicted_label": label, "probabilities": probs.tolist(), "instructor_attention": A_i.tolist(), "learner_attention": A_l.tolist()})

            # DSMIL
            res_dsmil = "N/A"
            if dsmil_model:
                with torch.no_grad():
                    feats = features.to(device)
                    ins_scores, bag_prediction, A, _ = dsmil_model(feats)
                    max_prediction, _ = torch.max(ins_scores, 0)
                    score = 0.5 * torch.sigmoid(bag_prediction) + 0.5 * torch.sigmoid(max_prediction.view(1, -1))
                    probs = torch.softmax(score, dim=1).squeeze().cpu().numpy()
                    label = "LUAD" if np.argmax(probs) == 0 else "LUSC"
                    res_dsmil = f"{label} ({format_probs(probs)})"
                    save_log(osid, "dsmil", {"timestamp": timestamp, "slide_id": osid, "ground_truth": gt, "model": "DSMIL", "predicted_label": label, "probabilities": probs.tolist(), "i_classifier_scores": ins_scores.cpu().numpy().tolist(), "b_classifier_attention": A.squeeze().cpu().numpy().tolist()})

            results.append({"Slide": osid, "Ground Truth": gt, "ABMIL": res_abmil, "CEMIL": res_cemil, "DSMIL": res_dsmil})

    except Exception as e:
        return pd.DataFrame({"Error": [str(e)]})
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# UI
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
                label="Enter absolute paths or slide IDs (one per line)",
                placeholder="slide1.svs\n/path/to/slide2.svs",
                lines=5,
            )
            gr.Markdown(
                "> ✅ **Recommended** when running on the same server as the data. "
                "No upload required — files are processed directly from disk."
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
    demo.launch(server_name="0.0.0.0", server_port=7910, share=True)
