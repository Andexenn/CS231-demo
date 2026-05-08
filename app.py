import gradio as gr
import os
import shutil
import subprocess
import torch
import sys
import glob

# Setup Base Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "inference_model"))

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
    cemil_instructor, cemil_learner = load_cemil(cemil_instructor_ckpt, cemil_learner_ckpt, device=device)
    dsmil_model = load_dsmil(dsmil_ckpt, device=device)
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")

def format_probs(probs):
    """Format numpy probabilities nicely."""
    if isinstance(probs, list) or hasattr(probs, 'tolist'):
        probs = probs.tolist()
    return f"LUAD: {probs[0]:.4f}, LUSC: {probs[1]:.4f}"

def process_wsi(wsi_file):
    if wsi_file is None:
        return "No file uploaded.", "", ""

    extract_dir = os.path.join(BASE_DIR, "extract")
    prepath_dir = os.path.join(extract_dir, "PrePATH")

    if not os.path.exists(prepath_dir):
        return "PrePATH directory not found.", "", ""

    # Setup isolated temp directories for this processing run
    temp_dir = os.path.join(BASE_DIR, "temp_wsi_processing")
    os.makedirs(temp_dir, exist_ok=True)
    
    wsi_dir = os.path.join(temp_dir, "wsi")
    os.makedirs(wsi_dir, exist_ok=True)
    
    # We must ensure it has an .svs extension for PrePATH to find it automatically
    wsi_path = os.path.join(wsi_dir, "uploaded.svs")
    shutil.copy(wsi_file.name, wsi_path)

    patch_dir = os.path.join(temp_dir, "patches")
    csv_dir = os.path.join(temp_dir, "csv")
    feat_dir = os.path.join(temp_dir, "feats")
    
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(feat_dir, exist_ok=True)

    try:
        # 1. Patching Phase
        print("Running PrePATH patching...")
        cmd_patch = [
            sys.executable, "create_patches_fp.py",
            "--source", wsi_dir,
            "--save_dir", patch_dir,
            "--patch_size", "256",
            "--preset", "tcga.csv",
            "--patch_level", "0",
            "--wsi_format", "svs",
            "--seg", "--patch"
        ]
        subprocess.run(cmd_patch, cwd=prepath_dir, check=True)

        h5_dir = os.path.join(patch_dir, "patches")

        # 2. Generate CSV for Extraction
        print("Generating CSV...")
        cmd_csv = [
            sys.executable, "scripts/extract_feature/generate_csv.py",
            "--h5_dir", h5_dir,
            "--num", "1",
            "--root", csv_dir
        ]
        subprocess.run(cmd_csv, cwd=prepath_dir, check=True)

        # 3. Extract Features Phase
        print("Extracting features with ResNet50...")
        cmd_feat = [
            sys.executable, "extract_features_fp_fast.py",
            "--model", "resnet50",
            "--csv_path", os.path.join(csv_dir, "part_0.csv"),
            "--data_coors_dir", patch_dir,
            "--data_slide_dir", wsi_dir,
            "--feat_dir", feat_dir,
            "--ignore_partial", "yes",
            "--batch_size", "128",
            "--datatype", "auto",
            "--slide_ext", ".svs",
            "--save_storage", "yes"
        ]
        subprocess.run(cmd_feat, cwd=prepath_dir, check=True)

        # 4. Load resulting features
        pt_files = glob.glob(os.path.join(feat_dir, "pt_files", "resnet50", "*.pt"))
        if not pt_files:
            return "Feature extraction failed: No .pt file found.", "", ""
        
        features = torch.load(pt_files[0], map_location=device)
        if not isinstance(features, torch.Tensor):
            features = torch.tensor(features)
        features = features.float()

        print(f"Extracted Features shape: {features.shape}")
        
        if features.dim() > 2:
            features = features.squeeze(0)

        class_map = {0: "LUAD", 1: "LUSC"}

        # 5. Run inference across the 3 models
        res_abmil = "Model not loaded"
        if abmil_model:
            pred_abmil, probs_abmil, _ = predict_abmil(abmil_model, features, device=device)
            res_abmil = f"{class_map[pred_abmil]} ({format_probs(probs_abmil)})"

        res_cemil = "Model not loaded"
        if cemil_instructor and cemil_learner:
            pred_cemil, probs_cemil, _, _, _ = predict_cemil(cemil_instructor, cemil_learner, features, device=device)
            res_cemil = f"{class_map[pred_cemil]} ({format_probs(probs_cemil)})"

        res_dsmil = "Model not loaded"
        if dsmil_model:
            pred_dsmil, probs_dsmil, _ = predict_dsmil(dsmil_model, features, device=device)
            res_dsmil = f"{class_map[pred_dsmil]} ({format_probs(probs_dsmil)})"

    except subprocess.CalledProcessError as e:
        return f"Extraction process failed: {e}", "", ""
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error occurred: {str(e)}", "", ""
    finally:
        # Clean up temporary processing directories to save space
        shutil.rmtree(temp_dir, ignore_errors=True)

    return res_abmil, res_cemil, res_dsmil

# Gradio UI Definition
with gr.Blocks(title="WSI Classifier") as demo:
    gr.Markdown("# WSI Feature Extraction and Classification Demo")
    gr.Markdown(
        "Upload a Whole Slide Image (`.svs`) below. The system will automatically:\n"
        "1. Identify tissue patches using **PrePATH**.\n"
        "2. Extract visual features using a pre-trained **ResNet50**.\n"
        "3. Run the extracted features through three advanced Multiple Instance Learning models: **ABMIL**, **CEMIL**, and **DSMIL**.\n"
        "4. Display the resulting classification predictions (**LUAD** or **LUSC**)."
    )
    
    with gr.Row():
        wsi_input = gr.File(label="Upload .svs WSI", file_types=[".svs"])
        
    run_btn = gr.Button("Extract Features & Predict", variant="primary")
        
    with gr.Row():
        out_abmil = gr.Textbox(label="ABMIL Prediction")
        out_cemil = gr.Textbox(label="CEMIL Prediction")
        out_dsmil = gr.Textbox(label="DSMIL Prediction")

    run_btn.click(fn=process_wsi, inputs=[wsi_input], outputs=[out_abmil, out_cemil, out_dsmil])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
