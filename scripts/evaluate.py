import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from data_preprocess import test_dataset, custom_collate_test
from network import VisionOnlyNetwork, InertialOnlyNetwork, VisualInertialNetwork
from loss_fn import TranslationRotationLoss
# from train import vision_loss_fn, inertial_loss_fn, visual_inertial_loss_fn

def evaluate_model(model, loss_fn, dataloader, device):
    model.eval()
    total_loss = 0
    total_samples = 0
    all_gt_positions = []
    all_pred_positions = []

    with torch.no_grad():
        for data in dataloader:
            if data is None:
                continue
            # img1, img2, imu_data, gt_rel_pose = [d.to(device) for d in data]

            vision_data, inertial_data, visual_inertial_data = data
            img1, img2, gt_rel_pose = vision_data
            imu_data = inertial_data

            print(f"img1 type: {type(img1)}")
            print(f"img2 type: {type(img2)}")
            print(f"imu_data type: {type(imu_data)}")
            print(f"gt_rel_pose type: {type(gt_rel_pose)}")
            
            img1, img2 = img1.to(device), img2.to(device)
            imu_data, gt_rel_pose = imu_data.to(device), gt_rel_pose.to(device)

            if isinstance(model, VisionOnlyNetwork):
                pred_rel_pose = model(img1, img2)
            elif isinstance(model, InertialOnlyNetwork):
                pred_rel_pose = model(imu_data)
            elif isinstance(model, VisualInertialNetwork):
                pred_rel_pose = model(img1, img2, imu_data)
            else:
                raise ValueError("Unknown model type")


            loss = loss_fn(pred_rel_pose, gt_rel_pose)
            total_loss += loss.item() * gt_rel_pose.size(0)
            total_samples += gt_rel_pose.size(0)

            gt_positions, pred_positions = integrate_poses(gt_rel_pose, pred_rel_pose, device)
            all_gt_positions.extend(gt_positions)
            all_pred_positions.extend(pred_positions)

    mean_loss = total_loss / total_samples
    return mean_loss, all_gt_positions, all_pred_positions

def integrate_poses(gt_rel_pose, pred_rel_pose, device):
    gt_positions = [torch.zeros((3,), device=device)]  # Initial position at (0, 0, 0)
    pred_positions = [torch.zeros((3,), device=device)]

    for i in range(len(gt_rel_pose)):
        gt_positions.append(gt_positions[-1] + gt_rel_pose[i, :3].to(device))
        pred_positions.append(pred_positions[-1] + pred_rel_pose[i, :3].to(device))

    return gt_positions, pred_positions

def plot_trajectory(gt_positions, pred_positions, title):
    gt_positions = torch.stack(gt_positions).cpu().numpy()
    pred_positions = torch.stack(pred_positions).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.plot(gt_positions[:, 0], gt_positions[:, 1], label="Ground Truth")
    plt.plot(pred_positions[:, 0], pred_positions[:, 1], label="Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title(title)
    plt.grid()
    plt.show()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load trained models
vision_model = VisionOnlyNetwork().to(device)
inertial_model = InertialOnlyNetwork().to(device)
visual_inertial_model = VisualInertialNetwork().to(device)

vision_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/vision_model_epoch_2.pth"))
inertial_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/inertial_model/inertial_model_epoch_2.pth"))
visual_inertial_model.load_state_dict(torch.load("/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/visual_inertial_model/visual_inertial_model_epoch_2.pth"))

# Loss functions
vision_loss_fn = TranslationRotationLoss()
inertial_loss_fn = TranslationRotationLoss()
visual_inertial_loss_fn = TranslationRotationLoss()
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_test)
# Evaluate models
vision_loss, vision_gt_positions, vision_pred_positions = evaluate_model(vision_model, vision_loss_fn, test_dataloader, device)
inertial_loss, inertial_gt_positions, inertial_pred_positions = evaluate_model(inertial_model, inertial_loss_fn, test_dataloader, device)
visual_inertial_loss, visual_inertial_gt_positions, visual_inertial_pred_positions = evaluate_model(visual_inertial_model, visual_inertial_loss_fn, test_dataloader, device)

print("Vision Only Loss:", vision_loss)
print("Inertial Only Loss:", inertial_loss)
print("Visual Inertial Loss:", visual_inertial_loss)

# Plot trajectories
plot_trajectory(vision_gt_positions, vision_pred_positions, "Vision Only Network")
plot_trajectory(inertial_gt_positions, inertial_pred_positions, "Inertial Only Network")
plot_trajectory(visual_inertial_gt_positions, visual_inertial_pred_positions, "Visual-Inertial Network")