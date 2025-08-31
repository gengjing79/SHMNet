import torch
from torch import nn
class EMAWeightGenerator(nn.Module):
    def __init__(self, feature_dim, num_experts=1, ema_decay=0.999):
        super().__init__()
        self.ema_decay = ema_decay
        self.register_buffer('ema_weights', torch.tensor([0.5, 0.5]))
        self.alpha = nn.Parameter(torch.tensor(0.7))

        self.attention = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Linear(64, num_experts),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.alpha * self.attention(x)

    def update_ema(self, current_weights):
        with torch.no_grad():
            self.ema_weights.data = (
                    self.ema_decay * self.ema_weights +
                    (1 - self.ema_decay) * current_weights.mean(dim=0)
            )


