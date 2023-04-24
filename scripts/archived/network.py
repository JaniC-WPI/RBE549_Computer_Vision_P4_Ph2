import torch
import torch.nn as nn

class VI_Network(nn.Module):
    def __init__(self):
        super(VI_Network, self).__init__()

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
            nn.Flatten()
        )

        self.lstm = nn.LSTM(
            input_size=6,
            hidden_size=128,
            num_layers=1,
            batch_first=True
        )

        # self.fc = nn.Linear(128 * 2, 6)
        self.fc = nn.Linear(1443968, 7)

    def forward(self, img1, img2, imu_seq):
        img_features1 = self.cnn(img1)
        img_features2 = self.cnn(img2)

        lstm_out, _ = self.lstm(imu_seq)
        lstm_features = lstm_out[:, -1, :]

        features = torch.cat((img_features1, img_features2, lstm_features), dim=-1)
        # print("features shape:", features.shape)
        output = self.fc(features)

        return output