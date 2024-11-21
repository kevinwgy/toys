import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

model_name  = "../model_scripted.pt";
input_name  = "../DataPrep/validation_data.txt";
output_name = "test_results.txt";

#--------------------------
# Load model
#--------------------------
model = torch.jit.load(model_name)
model.eval();

# ---------------------------------------------
# Load training data and validation/test data
# ---------------------------------------------
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, filename):
    self.filename = filename;
    rawdata = np.genfromtxt(self.filename);
    ncol = rawdata.shape[1];
    self.X = rawdata[:, 0:3];
    self.F = rawdata[:,ncol-6:ncol];
  def __len__(self):
    return len(self.F);
  def __getitem__(self, idx):
    return self.X[idx,:], self.F[idx,:];
  #def plot(self):
  #  return plt.scatter(self.X[:,0], self.X[:,1]);

input_data = MyDataset(input_name);

# ---------------------------------------------
# Check conservation
# ---------------------------------------------
# write the predicted results to a file
with open(output_name, "w") as f:
  for x, y0 in input_data:
    x_torch = torch.from_numpy(x);
    y_torch = model(x_torch.float());
    y = y_torch.detach().numpy();

    error_2norm = np.linalg.norm(y-y0, ord=2);
    conservation_error = (y[0] + y[1] + y[2] - y[3] - y[4] - y[5])/x[0] - 1.0;

    f.write(f"{error_2norm:16.8e}");
    f.write(f"{conservation_error:16.8e}");
    f.write("\n");
