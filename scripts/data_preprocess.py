import os
import glob
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset


class EuRoCDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.image_files = sorted(glob.glob(os.path.join(root_dir, 'mav0', 'cam0', 'data', '*.png')))
        self.imu_data = pd.read_csv(os.path.join(root_dir, 'mav0', 'imu0', 'data.csv'))
        self.ground_truth = pd.read_csv(os.path.join(root_dir, 'mav0', 'state_groundtruth_estimate0', 'data.csv'))

    def __len__(self):
        return len(self.image_files) - 1

    def __getitem__(self, idx):
        img1 = cv2.imread(self.image_files[idx], cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(self.image_files[idx + 1], cv2.IMREAD_GRAYSCALE)
        
        # Reshape images to (C, H, W) and normalize
        img1 = np.expand_dims(img1, axis=0) / 255.0
        img2 = np.expand_dims(img2, axis=0) / 255.0
        
        # Get timestamps of images and find IMU measurements between them
        img1_timestamp = int(os.path.basename(self.image_files[idx]).split('.')[0])
        img2_timestamp = int(os.path.basename(self.image_files[idx + 1]).split('.')[0])
        
        imu_seq = self.imu_data[(self.imu_data.timestamp >= img1_timestamp) & (self.imu_data.timestamp <= img2_timestamp)]
        imu_seq = imu_seq.iloc[:, 1:].values

        # Get ground truth pose for img1 and img2
        gt_pose1 = self.ground_truth[self.ground_truth.timestamp == img1_timestamp]
        gt_pose2 = self.ground_truth[self.ground_truth.timestamp == img2_timestamp]

        gt_pose1 = gt_pose1.iloc[0, 1:].values
        gt_pose2 = gt_pose2.iloc[0, 1:].values
        
        # Calculate relative pose between img1 and img2
        gt_rel_pose = np.hstack((gt_pose2[:3] - gt_pose1[:3], gt_pose2[3:] - gt_pose1[3:]))

        if self.transform:
            img1, img2, imu_seq, gt_rel_pose = self.transform((img1, img2, imu_seq, gt_rel_pose))

        return img1, img2, imu_seq, gt_rel_pose


# Create a dataset instance for each sequence
datasets = []
for seq in ['MH_02_easy', 'MH_03_medium', 'MH_04_difficult', 'MH_05_difficult']:
    datasets.append(EuRoCDataset(os.path.join('path/to/your/data', seq)))

# Concatenate all datasets
full_dataset = ConcatDataset(datasets)

# Split into training, validation, and test sets
total_size = len(full_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(full_dataset, [train_size, val_size, test_size])

# Create data loaders
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)