import torch
import torch.nn as nn


class VisionLSTM_Network(nn.Module):
    def __init__(self):
        super(VisionLSTM_Network, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(output_size=(6, 8)),
            nn.Flatten()
        )

        cnn_output_size = 128 * 6 * 8  # Calculate the output size of the last CNN layer (C * H * W)
        self.lstm = nn.LSTM(
            input_size=cnn_output_size,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(128, 7)

    def forward(self, img1, img2):
        cnn_img1 = self.cnn(img1)
        cnn_img2 = self.cnn(img2)

        cnn_features = torch.cat((cnn_img1, cnn_img2), dim=1).view(img1.shape[0], 2, -1)
        # cnn_features = cnn_features.unsqueeze(1)  # Add a sequence length dimension

        lstm_out, _ = self.lstm(cnn_features)
        lstm_features = lstm_out[:, -1, :]

        output = self.fc(lstm_features)

        return output