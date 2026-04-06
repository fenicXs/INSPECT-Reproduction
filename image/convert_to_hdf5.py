import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def convert_npy_to_hdf5(input_dir, output_path, metadata_path):
    # Read metadata to get mapping between impression_id and image_id
    df_metadata = pd.read_csv(metadata_path)

    # Create HDF5 file
    with h5py.File(output_path, 'w') as hdf5_file:
        # Get all NPY files
        npy_files = list(Path(input_dir).glob('*.npy'))
        print(f"Found {len(npy_files)} NPY files")

        for npy_file in npy_files:
            try:
                # Get image_id from filename (filename is already the image_id without .nii.gz)
                image_id = npy_file.stem

                # Load features
                features = np.load(npy_file)

                # Ensure features are 2D (num_slices x feature_dim)
                if len(features.shape) == 1:
                    features = features.reshape(1, -1)

                # Save to HDF5 using the image_id as key
                hdf5_file.create_dataset(image_id, data=features, dtype='float32')
            except Exception as e:
                print(f"Error processing {npy_file}: {str(e)}")
                continue

    print(f"HDF5 file created at: {output_path}")

if __name__ == "__main__":
    input_dir = "/scratch/pkrish52/INSPECT/output/ct_features"
    output_path = "/scratch/pkrish52/INSPECT/output/ct_features/features.hdf5"
    metadata_path = "/scratch/pkrish52/INSPECT/data/image_pipeline/Final_metadata.csv"

    convert_npy_to_hdf5(input_dir, output_path, metadata_path)
