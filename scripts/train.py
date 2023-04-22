import torch
import torch.nn as nn
from network import VI_Network
from data_preprocess import train_dataloader, val_dataloader


def pose_loss(predicted_pose, gt_pose):
    mse_loss = nn.MSELoss()
    position_loss = mse_loss(predicted_pose[:, :3], gt_pose[:, :3])
    orientation_loss = mse_loss(predicted_pose[:, 3:], gt_pose[:, 3:])
    
    return position_loss + orientation_loss


# Set your hyperparameters
learning_rate = 1e-4
num_epochs = 50

# Create a network instance
network = VI_Network()

# Move the network to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
network.to(device)

# Create an optimizer
optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    for batch in train_dataloader:
        img1, img2, imu_seq, gt_rel_pose = batch
        if img1 is None:
            continue

        # Move tensors to the device
        img1 = img1.to(device)
        img2 = img2.to(device)
        imu_seq = imu_seq.to(device)
        gt_rel_pose = gt_rel_pose.to(device)

        optimizer.zero_grad()
        predicted_pose = network(img1, img2, imu_seq)

        print("predicted_pose shape:", predicted_pose.shape)
        print("gt_rel_pose shape:", gt_rel_pose.shape)

        loss = pose_loss(predicted_pose, gt_rel_pose)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print average loss per epoch
    avg_loss = running_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

    # Evaluate on the validation set
    if (epoch + 1) % 10 == 0:
        network.eval()
        with torch.no_grad():
            val_loss = 0.0
            for data in val_dataloader:
                img1, img2, imu_seq, gt_pose = data
                img1, img2, imu_seq, gt_pose = img1.to(device), img2.to(device), imu_seq.to(device), gt_pose.to(device)

                predicted_pose = network(img1, img2, imu_seq)
                loss = pose_loss(predicted_pose, gt_pose)

                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_dataloader)
            print(f"Validation Loss: {avg_val_loss:.4f}")
        network.train()