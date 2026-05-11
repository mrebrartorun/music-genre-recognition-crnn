"""
model_improved.py
=================
MYZ307E — Music Genre Recognition
Author  : Mesut Anlak
Module  : Improved CRNN — 5-block ResNet front-end + 2-layer BiLSTM + Temporal Attention Pooling

Drop-in replacement for the baseline CRNN.
Input shape  : (batch, 1, 128, 130)   — same as baseline
Output shape : (batch, 10)            — same as baseline (10 genres, raw logits)

Architecture overview
---------------------
1. ResNet Front-end  : 5 residual blocks (increasing channels: 64→128→128→256→256)
                       Each block: Conv→BN→ELU → Conv→BN + skip-connection → ELU → MaxPool → Dropout
2. Sequence reshape  : collapse frequency dim into channels, keep time dim as sequence axis
3. BiLSTM            : 2-layer Bidirectional LSTM (hidden=256 → output=512 per step)
4. Temporal Attention: learnable softmax over time steps → weighted sum → context vector
5. Classifier        : Linear(512, 10)

Usage
-----
    from model_improved import ImprovedCRNN
    model = ImprovedCRNN(num_classes=10, dropout=0.3)
    logits = model(x)   # x: (B, 1, 128, 130)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Residual Block (2 conv layers + identity / projection skip)
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    """
    One residual block:
        x → Conv2d → BN → ELU → Dropout → Conv2d → BN → (+x_skip) → ELU → MaxPool
    If in_channels != out_channels a 1×1 projection is used for the skip.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 pool_size: tuple = (2, 2), dropout: float = 0.3):
        super().__init__()

        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)

        self.drop  = nn.Dropout2d(p=dropout)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        # Skip connection (projection if channel mismatch)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        self.activation = nn.ELU(inplace=True)
        self.pool = nn.MaxPool2d(pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.skip(x)

        out = self.activation(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))

        out = self.activation(out + identity)   # residual add
        out = self.pool(out)
        return out


# ---------------------------------------------------------------------------
# Helper: Temporal Attention Pooling
# ---------------------------------------------------------------------------
class TemporalAttention(nn.Module):
    """
    Learns a scalar energy for each time step, normalises with softmax,
    and returns the weighted sum over the sequence dimension.

    Input  : (B, T, H)   — BiLSTM output
    Output : (B, H)      — attended context vector
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Single linear layer → scalar energy per time step
        self.attention = nn.Linear(hidden_size, 1, bias=True)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        # lstm_out : (B, T, H)
        energies = self.attention(lstm_out)          # (B, T, 1)
        weights  = F.softmax(energies, dim=1)        # (B, T, 1)  — over time
        context  = (weights * lstm_out).sum(dim=1)   # (B, H)
        return context


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------
class ImprovedCRNN(nn.Module):
    """
    Improved CRNN for music genre classification.

    Parameters
    ----------
    num_classes : int   — number of output classes (default: 10 for GTZAN)
    dropout     : float — dropout probability used in ResBlocks and LSTM
    lstm_hidden : int   — hidden size of each LSTM direction (default: 256)
    """

    def __init__(self,
                 num_classes: int = 10,
                 dropout:     float = 0.3,
                 lstm_hidden: int = 256):
        super().__init__()

        # ------------------------------------------------------------------
        # 1. 5-Block ResNet Front-end
        #    Input : (B, 1,   128, 130)
        #    After block 1 : (B, 64,  64, 65)
        #    After block 2 : (B, 128, 32, 32)
        #    After block 3 : (B, 128, 16, 16)
        #    After block 4 : (B, 256,  8,  8)
        #    After block 5 : (B, 256,  4,  4)
        # ------------------------------------------------------------------
        self.resnet = nn.Sequential(
            ResBlock(  1,  64, pool_size=(2, 2), dropout=dropout),   # block 1
            ResBlock( 64, 128, pool_size=(2, 2), dropout=dropout),   # block 2
            ResBlock(128, 128, pool_size=(2, 2), dropout=dropout),   # block 3
            ResBlock(128, 256, pool_size=(2, 2), dropout=dropout),   # block 4
            ResBlock(256, 256, pool_size=(2, 2), dropout=dropout),   # block 5
        )

        # After 5 MaxPool2d(2,2) on (128,130):
        #   freq : 128 / 2^5 = 4
        #   time : 130 / 2^5 = 4   (floor division at each step)
        # Sequence length T = time_frames_after_pooling = 4
        # Feature size per step = channels × freq = 256 × 4 = 1024

        cnn_out_channels = 256
        cnn_freq_bins    = 4    # 128 >> 5
        lstm_input_size  = cnn_out_channels * cnn_freq_bins   # 1024

        # ------------------------------------------------------------------
        # 2. 2-layer Bidirectional LSTM
        #    input  : (T, B, 1024)
        #    output : (T, B, lstm_hidden*2)  = (T, B, 512)
        # ------------------------------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,          # expect (B, T, input_size)
            bidirectional=True,
            dropout=dropout if dropout > 0 else 0,
        )

        lstm_output_size = lstm_hidden * 2   # bidirectional

        # ------------------------------------------------------------------
        # 3. Temporal Attention Pooling
        # ------------------------------------------------------------------
        self.attention = TemporalAttention(lstm_output_size)

        # ------------------------------------------------------------------
        # 4. Classifier
        # ------------------------------------------------------------------
        self.dropout    = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(lstm_output_size, num_classes)

        # Weight initialisation
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. ResNet front-end (Giriş: (B, 1, 128, 130) -> Çıkış: (B, 256, 4, 4))
        x = self.resnet(x) 

        # 2. Sequence Reshape (Frekanstaki bilgiyi kanallara katlayıp zamanı eksenine alıyoruz)
        # (B, C, F, T) -> (B, T, C, F)
        x = x.permute(0, 3, 1, 2).contiguous()
        batch_size, time_steps, channels, freq = x.size()
        
        # (B, T, 256*4) -> (B, 4, 1024)
        x = x.view(batch_size, time_steps, -1)

        # 3. BiLSTM (Giriş: (B, 4, 1024) -> Çıkış: (B, 4, 512))
        lstm_out, _ = self.bilstm(x)

        # 4. Temporal Attention Pooling (Zaman adımları üzerinde ağırlıklı toplam)
        # Çıkış: (B, 512)
        context = self.attention(lstm_out)

        # 5. Classify
        out = self.dropout(context)
        logits = self.classifier(out)
        return logits

# ---------------------------------------------------------------------------
# Convenience factory (mirrors naming convention used in baseline)
# ---------------------------------------------------------------------------
def build_model(num_classes: int = 10,
                dropout:     float = 0.3,
                lstm_hidden: int = 256) -> ImprovedCRNN:
    """Return an initialised ImprovedCRNN ready for training."""
    return ImprovedCRNN(num_classes=num_classes,
                        dropout=dropout,
                        lstm_hidden=lstm_hidden)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = build_model()
    print(model)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters : {total_params:,}")

    # Dummy forward pass
    dummy = torch.zeros(4, 1, 128, 130)   # batch=4, channels=1, freq=128, time=130
    with torch.no_grad():
        out = model(dummy)
    print(f"Input shape  : {dummy.shape}")
    print(f"Output shape : {out.shape}")    # expected: (4, 10)
    assert out.shape == (4, 10), "Output shape mismatch!"
    print("\n✓ Forward pass OK")
