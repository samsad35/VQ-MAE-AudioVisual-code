import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(84, 8)

    def forward(self):
        pass


def size_model(model: torch.nn.Module, text: str = ""):
    size = 0
    for param in model.parameters():
        if param.data.is_floating_point():
            size += param.numel() * torch.finfo(param.data.dtype).bits
        else:
            size += param.numel() * torch.iinfo(param.data.dtype).bits

    number_of_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"model size of {text}: {size} / bit | {size / 8e6:.2f} / MB | {number_of_parameters} / param.")


if __name__ == '__main__':
    model = Net()
    size_model(model)
