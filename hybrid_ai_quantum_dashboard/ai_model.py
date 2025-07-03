try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError:
    raise ImportError("PyTorch is not installed. Please install it with `pip install torch`.")

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(16 * 112 * 112, 10)
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    model = SimpleCNN()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
