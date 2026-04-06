"""Featurize NIfTI CT volumes slice-by-slice, loading each volume only once.

This is more efficient than the standard pipeline which was designed for DICOM
(one file per slice). With NIfTI (all slices in one file), we load each volume
once and process all its slices in batches.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import nibabel as nib
import pandas as pd
import h5py
import timm
import cv2
from tqdm import tqdm
from pathlib import Path
from torchvision import transforms

# --- Config ---
CHECKPOINT_PATH = "/scratch/pkrish52/INSPECT/resnetv2_ct/resnetv2_ct.ckpt"
METADATA_CSV = "/scratch/pkrish52/INSPECT/data/image_pipeline/Final_metadata.csv"
CTPA_DIR = "/scratch/pkrish52/INSPECT/data/CTPA/inspectamultimodaldatasetforpulmonaryembolismdiagnosisandprog-3/full/CTPA"
OUTPUT_HDF5 = "/scratch/pkrish52/INSPECT/output/ct_features/features.hdf5"
BATCH_SIZE = 64
RESIZE = 256
CROP = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Identity(nn.Module):
    def forward(self, x):
        return x


def load_model():
    """Load pre-trained ResNetV2-101x3 with CT weights."""
    model = timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=False)
    model.head.fc = Identity()

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
    state = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()}
    msg = model.load_state_dict(state, strict=False)
    print(f"Model loaded. Missing: {len(msg.missing_keys)}, Unexpected: {len(msg.unexpected_keys)}")

    model = model.to(DEVICE)
    model.eval()
    return model


def windowing(pixel_array, window_center, window_width):
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    pixel_array = np.clip(pixel_array, lower, upper)
    return (pixel_array - lower) / (upper - lower)


def get_transform():
    return transforms.Compose([
        transforms.Resize(RESIZE),
        transforms.CenterCrop(CROP),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def process_volume(nifti_path, model, transform):
    """Load a NIfTI volume, extract features for all slices."""
    img = nib.load(nifti_path)
    volume = img.get_fdata()  # (H, W, num_slices)

    if len(volume.shape) == 4:
        volume = volume[:, :, :, 0]

    num_slices = volume.shape[2]
    all_features = []

    # Process in batches
    for start in range(0, num_slices, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_slices)
        batch_tensors = []

        for s in range(start, end):
            slice_2d = volume[:, :, s]

            # Resize if needed
            if slice_2d.shape[0] != RESIZE or slice_2d.shape[1] != RESIZE:
                slice_2d = cv2.resize(slice_2d, (RESIZE, RESIZE), interpolation=cv2.INTER_AREA)

            # Apply 3-channel windowing (LUNG, PE, MEDIASTINAL)
            lung = windowing(slice_2d, -600, 1500)
            pe = windowing(slice_2d, 400, 1000)
            mediastinal = windowing(slice_2d, 40, 400)
            ct_3ch = np.stack([lung, pe, mediastinal])  # (3, H, W)

            # Convert to PIL and apply transforms
            ct_3ch = np.transpose(ct_3ch, (1, 2, 0))  # (H, W, 3)
            from PIL import Image
            pil_img = Image.fromarray(np.uint8(ct_3ch * 255))
            tensor = transform(pil_img)
            batch_tensors.append(tensor)

        batch = torch.stack(batch_tensors).to(DEVICE)

        with torch.no_grad(), torch.cuda.amp.autocast():
            features = model(batch)  # (batch, 6144)

        all_features.append(features.cpu().numpy())

    return np.concatenate(all_features, axis=0)  # (num_slices, 6144)


def main():
    os.makedirs(os.path.dirname(OUTPUT_HDF5), exist_ok=True)

    # Load metadata
    df = pd.read_csv(METADATA_CSV)
    print(f"Processing {len(df)} volumes")

    # Load model
    model = load_model()
    transform = get_transform()

    # Check for existing progress
    processed = set()
    if os.path.exists(OUTPUT_HDF5):
        with h5py.File(OUTPUT_HDF5, "r") as f:
            processed = set(f.keys())
        print(f"Resuming: {len(processed)} already processed")

    # Open HDF5 for writing (append mode)
    with h5py.File(OUTPUT_HDF5, "a") as hdf5_file:
        errors = 0
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            image_id = row["image_id"]
            key = image_id.replace(".nii.gz", "") if str(image_id).endswith(".nii.gz") else str(image_id)

            # Skip if already processed
            if key in processed:
                continue

            nifti_path = os.path.join(CTPA_DIR, image_id if str(image_id).endswith(".nii.gz") else f"{image_id}.nii.gz")

            try:
                features = process_volume(nifti_path, model, transform)
                hdf5_file.create_dataset(key, data=features, dtype="float32")

                if idx % 100 == 0:
                    hdf5_file.flush()
            except Exception as e:
                errors += 1
                print(f"Error processing {image_id}: {e}")
                continue

    print(f"\nDone! {len(df) - errors} volumes processed, {errors} errors")
    print(f"Features saved to {OUTPUT_HDF5}")


if __name__ == "__main__":
    main()
