import matplotlib.pyplot as plt
import numpy as np

def plot_losses(train_losses, val_losses, test_losses, val_epochs):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Training")
    plt.plot(val_epochs, val_losses, label="Validation")

    test_label_added = False
    for test_loss in test_losses:
        if not test_label_added:
            plt.axhline(y=test_loss, color='r', linestyle='--', label="Test")
            test_label_added = True
        else:
            plt.axhline(y=test_loss, color='r', linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def plot_pose_comparison(pred_poses_list, gt_poses_list):
    pred_positions = []
    gt_positions = []

    for pred_pose, gt_pose in zip(pred_poses_list, gt_poses_list):
        pred_positions.append(pred_pose[:3])
        gt_positions.append(gt_pose[:3])

    # pred_positions = np.array(pred_positions)
    # gt_positions = np.array(gt_positions)

    plt.scatter(gt_positions[:, 0], gt_positions[:, 1], marker='o', label='Ground Truth')
    plt.scatter(pred_positions[:, 0], pred_positions[:, 1], marker='.', label='Prediction')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.title('Ground Truth vs. Predicted Positions')
    plt.show()