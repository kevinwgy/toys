import os
import torch
from torch import nn

device = "cpu"; #("cuda" if torch.cuda.is_available() else "cpu");
print(f"Using {device} device");

class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 4),
      nn.ReLU(),
      nn.Linear(4, 4),
      nn.ReLU(),
      nn.Linear(4, 1),
    )

  def forward(self, x):
    return self.layers(x)

model = NeuralNetwork().to(device);
print(model);

X = torch.rand(1, 2); #one input, 2 dimensions (x,y)
Y = model(X);
Y = Y.to('cpu');
print(f"Prediction: y({X[0,0]},{X[0,1]})={Y.detach().numpy()[0,0]}")



