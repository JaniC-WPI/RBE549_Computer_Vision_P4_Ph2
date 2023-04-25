import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch

def rmse(gt, pred):
    mse = np.mean((gt - pred) ** 2)
    return np.sqrt(mse)

def evaluate(model, dataloader, device):
    model.eval()
    with torch.no_grad():
        gt_positions = []
        pred_positions = []

        for batch in dataloader:
            img_seq, gt_rel_pose = batch
            img_seq, gt_rel_pose = img_seq.to(device), gt_rel_pose.to(device)

            pred_rel_pose = model(img_seq)
            gt_positions.append(gt_rel_pose[:, :3].cpu().numpy())
            pred_positions.append(pred_rel_pose[:, :3].cpu().numpy())

        gt_positions = np.concatenate(gt_positions, axis=0)
        pred_positions = np.concatenate(pred_positions, axis=0)

        # Calculate the error metrics
        position_rmse = rmse(gt_positions, pred_positions)
        print(f"Position RMSE: {position_rmse:.3f}")

        # Plot the ground truth and predicted trajectories
        plt.figure()
        plt.plot(gt_positions[:, 0], gt_positions[:, 1], label='Ground Truth', color='blue')
        plt.plot(pred_positions[:, 0], pred_positions[:, 1], label='Predicted', color='red')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.title('Trajectory Comparison')
        plt.show()