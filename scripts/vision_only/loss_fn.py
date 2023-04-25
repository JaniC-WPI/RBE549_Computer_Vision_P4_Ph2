import torch.nn.functional as F
import torch
import torch.nn as nn

def loss_function(pred, gt):
    # Position (translation) loss
    position_pred = pred[:, :3]
    position_gt = gt[:, :3]
    position_loss = F.mse_loss(position_pred, position_gt)

    # Quaternion (rotation) loss
    quaternion_pred = pred[:, 3:]
    quaternion_gt = gt[:, 3:]
    rotation_loss = 1 - torch.sum(quaternion_pred * quaternion_gt, dim=1).pow(2)
    rotation_loss = torch.mean(rotation_loss)

    # Weighted sum of position and rotation losses
    position_weight = 1.0
    rotation_weight = 1.0
    total_loss = position_weight * position_loss + rotation_weight * rotation_loss

    return total_loss

def pose_loss(predicted_pose, gt_pose, alpha=0.5):
    mse_loss = nn.MSELoss()
    position_loss = mse_loss(predicted_pose[:, :3], gt_pose[:, :3])
    orientation_loss = mse_loss(predicted_pose[:, 3:], gt_pose[:, 3:])

    cosine_similarity_loss = nn.CosineSimilarity(dim=1)
    position_similarity = cosine_similarity_loss(predicted_pose[:, :3], gt_pose[:, :3])
    orientation_similarity = cosine_similarity_loss(predicted_pose[:, 3:], gt_pose[:, 3:])

    position_cosine_loss = 1 - position_similarity.mean()
    orientation_cosine_loss = 1 - orientation_similarity.mean()

    combined_position_loss = alpha * position_loss + (1 - alpha) * position_cosine_loss
    combined_orientation_loss = alpha * orientation_loss + (1 - alpha) * orientation_cosine_loss

    return combined_position_loss + combined_orientation_loss