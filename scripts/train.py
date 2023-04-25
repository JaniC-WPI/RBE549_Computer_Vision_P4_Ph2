import torch
import torch.nn as nn
# from data_preprocess_new import custom_collate_train, train_dataset
from data_preprocess import custom_collate, train_dataset
from loss_fn import TranslationRotationLoss, pose_loss
from network import VisionOnlyNetwork, InertialOnlyNetwork, VisualInertialNetwork
from torch.utils.data import Dataset, DataLoader

# Loss functions
inertial_loss_fn = TranslationRotationLoss()
visual_inertial_loss_fn = TranslationRotationLoss()


# Set your hyperparameters
learning_rate = 1e-4

# Create a network instance
vision_network = VisionOnlyNetwork()
inertial_network = InertialOnlyNetwork()
visual_inertial_network = VisualInertialNetwork()

# Move the network to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_network.to(device)
inertial_network.to(device)
visual_inertial_network.to(device)

# Create an optimizer
vision_optimizer = torch.optim.Adam(vision_network.parameters(), lr=learning_rate)
inertial_optimizer = torch.optim.Adam(inertial_network.parameters(), lr=learning_rate)
visual_inertial_optimizer = torch.optim.Adam(visual_inertial_network.parameters(), lr=learning_rate)

def train_vision_network(model, optimizer, img1, img2, gt_rel_pose):
    model.train()
    
    optimizer.zero_grad()

    img1 = img1.to(device)
    img2 = img2.to(device)
    gt_rel_pose = gt_rel_pose.to(device)
    
    # Forward pass
    pred_rel_pose = model(img1, img2)
    
    # Compute loss
    # loss = vision_loss_fn(pred_rel_pose, gt_rel_pose)
    loss = pose_loss(pred_rel_pose, gt_rel_pose, alpha=0.5)

    loss.backward()
    optimizer.step()

    return loss.item()

def train_inertial_network(model, optimizer, imu_data, gt_rel_pose):
    model.train()
    optimizer.zero_grad()

    imu_data = imu_data.to(device)
    gt_rel_pose = gt_rel_pose.to(device)

    pred_rel_pose = model(imu_data)
    loss = inertial_loss_fn(pred_rel_pose, gt_rel_pose)

    loss.backward()
    optimizer.step()

    return loss.item()


def train_visual_inertial_network(model, optimizer, img1, img2, imu_seq, gt_rel_pose):
    model.train()
    optimizer.zero_grad()

    img1 = img1.to(device)
    img2 = img2.to(device)
    imu_seq = imu_seq.to(device)
    gt_rel_pose = gt_rel_pose.to(device)

    pred_rel_pose = model(img1, img2, imu_seq)
    loss = visual_inertial_loss_fn(pred_rel_pose, gt_rel_pose)

    loss.backward()
    optimizer.step()

    return loss.item()

# Training loop
num_epochs = 50
batch_size = 10
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")

    # Training loop
    for i, data in enumerate(train_dataloader):
        if data is None:
            continue

        # vision_data, inertial_data, visual_inertial_data = data
        # img1, img2, gt_rel_pose_vision = vision_data
        # imu_seq, gt_rel_pose_inertial = inertial_data
        # img1_vi, img2_vi, imu_seq_vi, gt_rel_pose_vi = visual_inertial_data

        img1, img2, imu_seq, gt_rel_pose_vision = data

        if img1 is None or img2 is None:
            print("Skipping batch due to missing image(s)")
            continue

        train_vision_network(vision_network, vision_optimizer, img1, img2, gt_rel_pose_vision)
        # train_inertial_network(inertial_network, inertial_optimizer, imu_seq, gt_rel_pose_inertial)
        # train_visual_inertial_network(visual_inertial_network, visual_inertial_optimizer, img1_vi, img2_vi, imu_seq_vi, gt_rel_pose_vi)

    # Save the model weights after every epoch
    torch.save(vision_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/vision_model_epoch_{}.pth".format(epoch + 1))
    # torch.save(inertial_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/inertial_model/inertial_model_epoch_{}.pth".format(epoch + 1))
    # torch.save(visual_inertial_network.state_dict(), "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/visual_inertial_model/visual_inertial_model_epoch_{}.pth".format(epoch + 1))