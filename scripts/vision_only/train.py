import torch.optim as optim
import torch
import os
from data_preprocess import test_dataloader, train_dataloader, val_dataloader
from loss_fn import loss_function
from network import VisionLSTM_Network

def train(model, train_loader, device, epochs, learning_rate, save_path):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
    
        for i, batch in enumerate(train_loader, 1):
            img1, img2, gt_rel_pose = batch
            img1, img2, gt_rel_pose = img1.to(device), img2.to(device), gt_rel_pose.to(device)
    
            optimizer.zero_grad()
            pred_rel_pose = model(img1, img2)  # Pass both img1 and img2 as arguments
            loss = loss_function(pred_rel_pose, gt_rel_pose)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item()

        # Calculate average loss for this epoch
        epoch_loss = running_loss / i
        print(f"Epoch {epoch}/{epochs}, Loss: {epoch_loss:.4f}")

        # Save the model at the end of each epoch
        torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch}.pth"))

        # # Evaluate on the validation set
        # print("Evaluating on validation set...")
        # evaluate(model, val_loader)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionLSTM_Network().to(device)
epochs = 10
learning_rate = 1e-4
save_path = "/home/jc-merlab/RBE549_Computer_Vision_P4_Ph2/models/vision_model/"

os.makedirs(save_path, exist_ok=True)

train(model, train_dataloader, device, epochs, learning_rate, save_path)