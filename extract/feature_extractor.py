import os
import subprocess
import glob
import torch

def run_command(cmd):
    print(f"[INFO] Running: {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def main():
    # Setup Paths for LUAD
    DATA_COORS_DIR_LUAD = "/content/drive/MyDrive/CS231/tcga_dataset/luad_patches"
    H5_DIR_LUAD = os.path.join(DATA_COORS_DIR_LUAD, "patches")
    WSI_DIR_LUAD = "/content/drive/MyDrive/CS231/tcga_dataset/luad"
    FEAT_DIR_LUAD = "/content/drive/MyDrive/CS231/luad_feats"
    TASK_NAME_LUAD = "luad_resnet50"
    CSV_PATH_LUAD = f"csv/{TASK_NAME_LUAD}"

    os.makedirs(FEAT_DIR_LUAD, exist_ok=True)
    os.makedirs(CSV_PATH_LUAD, exist_ok=True)

    # Setup Paths for LUSC
    DATA_COORS_DIR_LUSC = "/content/drive/MyDrive/CS231/tcga_dataset/lusc_patches"
    H5_DIR_LUSC = os.path.join(DATA_COORS_DIR_LUSC, "patches")
    WSI_DIR_LUSC = "/content/drive/MyDrive/CS231/tcga_dataset/lusc"
    FEAT_DIR_LUSC = "/content/drive/MyDrive/CS231/lusc_feats"
    TASK_NAME_LUSC = "lusc_resnet50"
    CSV_PATH_LUSC = f"csv/{TASK_NAME_LUSC}"

    os.makedirs(FEAT_DIR_LUSC, exist_ok=True)
    os.makedirs(CSV_PATH_LUSC, exist_ok=True)

    print(f"PyTorch  : {torch.__version__}")
    print(f"CUDA     : {torch.cuda.is_available()}")

    # Change to PrePATH directory to run scripts
    prepath_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "PrePATH")
    if not os.path.exists(prepath_dir):
        print(f"[ERROR] PrePATH directory not found at {prepath_dir}. Please clone it first.")
        return
        
    os.chdir(prepath_dir)

    # 1. Generate CSV for LUSC (based on the notebook structure)
    print("\n--- Generating CSV ---")
    gen_csv_cmd = (
        f'python scripts/extract_feature/generate_csv.py '
        f'--h5_dir "{H5_DIR_LUSC}" '
        f'--num 1 '
        f'--root "{CSV_PATH_LUSC}"'
    )
    run_command(gen_csv_cmd)

    # 2. Extract Features for LUSC
    print("\n--- Extracting Features ---")
    extract_cmd = (
        f'python extract_features_fp_fast.py '
        f'--model "resnet50" '
        f'--csv_path "./{CSV_PATH_LUSC}/part_0.csv" '
        f'--data_coors_dir "{DATA_COORS_DIR_LUSC}" '
        f'--data_slide_dir "{WSI_DIR_LUSC}" '
        f'--feat_dir "{FEAT_DIR_LUSC}" '
        f'--ignore_partial yes '
        f'--batch_size 128 '
        f'--datatype auto '
        f'--slide_ext ".svs" '
        f'--save_storage "yes"'
    )
    run_command(extract_cmd)

    # 3. Verify extraction
    print("\n--- Verifying Extracted Features ---")
    pt_files = glob.glob(f"{FEAT_DIR_LUSC}/pt_files/resnet50/*.pt")
    if pt_files:
        features = torch.load(pt_files[0])
        print(f"Loaded: {pt_files[0]}")
        print(f"Feature Shape: {features.shape}") # Should be [num_patches, 1024]
    else:
        print("No features extracted yet.")

if __name__ == "__main__":
    main()
