import torch
import torch.nn as nn
from network import VisionOnlyNetwork
from data_preprocess import train_dataloader, val_dataloader, test_dataloader
from plot import plot_losses, plot_pose_comparison


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


# Set your hyperparameters
learning_rate = 1e-4
num_epochs = 50

# Create a network instance
network = VisionOnlyNetwork()

# Move the network to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)

# Create an optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

train_losses = []
val_losses = []
val_epochs = []
test_losses = []
pred_poses = []
gt_poses = []
train_pred_poses = []
train_gt_poses = []
test_pred_poses = []
test_gt_poses = []


# Training loop
for epoch in range(num_epochs):
    print("Num of Epoch:", epoch)
    running_loss = 0.0
    for batch in train_dataloader:
        if not batch: # skip empty batches
            continue
        img1, img2, imu_seq, gt_rel_pose = batch

        if img1 is None or img2 is None:
            continue

        # Move tensors to the device
        img1 = img1.to(device)
        img2 = img2.to(device)
        imu_seq = imu_seq.to(device)
        gt_rel_pose = gt_rel_pose.to(device)

        optimizer.zero_grad()
        predicted_pose = network(img1, img2)

        loss = pose_loss(predicted_pose, gt_rel_pose)
        train_pred_poses.append(predicted_pose.cpu().detach().numpy())
        train_gt_poses.append(gt_rel_pose.cpu().detach().numpy())
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss per epoch
    avg_loss = running_loss / len(train_dataloader)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    torch.save(network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/vision_model_epoch_{}.pth".format(epoch + 1))

    # # Evaluate on the validation set
    # if (epoch + 1) % 10 == 0:
    #     network.eval()
    #     with torch.no_grad():
    #         val_loss = 0.0
    #         for data in val_dataloader:
    #             img1, img2, imu_seq, gt_pose = data
    #             img1, img2, imu_seq, gt_pose = img1.to(device), img2.to(device), imu_seq.to(device), gt_pose.to(device)

    #             predicted_pose = network(img1, img2, imu_seq)
    #             loss = pose_loss(predicted_pose, gt_pose)

    #             val_loss += loss.item()
    #         avg_val_loss = val_loss / len(val_dataloader)
    #         val_losses.append(avg_val_loss)
    #         val_epochs.append(epoch + 1)  # Append the current epoch
    #         print(f"Validation Loss: {avg_val_loss:.4f}")
    #     network.train()

# Test loop
# network.eval()
# with torch.no_grad():
#     for data in test_dataloader:
#         img1, img2, imu_seq, gt_pose = data
#         img1, img2, imu_seq, gt_pose = img1.to(device), img2.to(device), imu_seq.to(device), gt_pose.to(device)

#         predicted_pose = network(img1, img2, imu_seq)
#         loss = pose_loss(predicted_pose, gt_pose)

#         test_losses.append(loss.item())
#         # pred_poses.append(predicted_pose.detach().cpu().numpy())
#         # gt_poses.append(gt_pose.detach().cpu().numpy())
#         test_pred_poses.append(predicted_pose.cpu().detach().numpy())
#         test_gt_poses.append(gt_pose.cpu().detach().numpy())

# network.train()

# print(f"Length of val_epochs: {len(val_epochs)}")
# print(f"Length of train_losses: {len(train_losses)}")
# print(f"Length of val_losses: {len(val_losses)}")
# print(f"Length of test_losses: {len(test_losses)}")

# plot_losses(train_losses, val_losses, test_losses, val_epochs)
# plot_pose_comparison(train_pred_poses, train_gt_poses)
# plot_pose_comparison(test_pred_poses, test_gt_poses)