import torch
import torch.nn as nn


class CRNNBaseline(nn.Module):
    """
    CRNN baseline adapted for preprocessed GTZAN data.

    Expected input shape:
        [batch_size, 1, 128, 130]

    Data meaning:
        128 = mel bins
        130 = time frames
    """

    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(0.1),
        )

        # Input: [B, 1, 128, 130]
        # After pooling 4 times:
        #   freq: 128 -> 64 -> 32 -> 16 -> 8
        #   time: 130 -> 65 -> 32 -> 16 -> 8
        # Output: [B, 128, 8, 8]
        # Time dimension is treated as the sequence length.
        # Features per time step = channels * freq = 128 * 8 = 1024

        self.gru1 = nn.GRU(
            input_size=1024,
            hidden_size=64,
            batch_first=True,
        )

        self.gru2 = nn.GRU(
            input_size=64,
            hidden_size=64,
            batch_first=True,
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # Current shape: [B, 128, 8, 8]
        # Convert to sequence: [B, time, features]
        x = x.permute(0, 3, 1, 2)                  # [B, 8, 128, 8]
        x = x.reshape(x.size(0), x.size(1), -1)    # [B, 8, 1024]

        x, _ = self.gru1(x)
        x, _ = self.gru2(x)

        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    model = CRNNBaseline(num_classes=10)
    dummy_input = torch.randn(4, 1, 128, 130)
    output = model(dummy_input)

    print(model)
    print("Input shape :", dummy_input.shape)
    print("Output shape:", output.shape)