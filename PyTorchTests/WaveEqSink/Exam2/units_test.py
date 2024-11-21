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

    error_2norm = np.linalg.norm(y-y0, ord=2) / np.linalg.norm(y0, ord=2);

    # Reduce a and beta_T by 10X, and predict results again!
    x2_torch = x_torch;
    x2_torch[0] /= 10.0;  #a
    x2_torch[2] /= 10.0;  #beta_T 
    y2_torch = model(x2_torch.float());
    y2 = y2_torch.detach().numpy();

    error2_2norm = np.linalg.norm(y2-y0/10.0, ord=2) / np.linalg.norm(y0/10.0, ord=2);

    #f.write(f"{x2_torch.detach().numpy()}");
    #f.write(f"{y2}");

    f.write(f"{error_2norm:16.8e}");
    f.write(f"{error2_2norm:16.8e}");
    f.write("\n");
