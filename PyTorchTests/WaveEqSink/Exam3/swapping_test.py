import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

model1_name  = "model_scripted.pt";
model2_name  = "model_scripted_swap.pt";
input_name  = "../DataPrep/validation_data.txt";
output_name = "test_results.txt";

#--------------------------
# Load model
#--------------------------
model1 = torch.jit.load(model1_name)
model2 = torch.jit.load(model2_name)
model1.eval();
model2.eval();

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
    return self.X[idx,:], self.F[idx];
  #def plot(self):
  #  return plt.scatter(self.X[:,0], self.X[:,1]);

input_data = MyDataset(input_name);

# ---------------------------------------------
# Check if two models yield the same result 
# ---------------------------------------------
# write the predicted results to a file
with open(output_name, "w") as f:
  for x, y0 in input_data:
    x_torch = torch.from_numpy(x);
    y_torch = model1(x_torch.float());
    y = y_torch.detach().numpy();

    error_2norm = np.linalg.norm(y-y0, ord=2) / np.linalg.norm(y0, ord=2);

    # swap x[0] and x[1], and call model 2 to make prediction
    x2_torch = x_torch;
    tmp = x2_torch[0].clone().detach();
    x2_torch[0] = x2_torch[1];
    x2_torch[1] = tmp;
    y2_torch = model2(x2_torch.float());
    y2 = y2_torch.detach().numpy();

    error2_2norm = np.linalg.norm(y2-y0, ord=2) / np.linalg.norm(y0, ord=2);

    diff_2norm = np.linalg.norm(y2-y, ord=2) / np.linalg.norm(y, ord=2);

    #f.write(f"{y}");
    #f.write(f"{y2}");
    f.write(f"{error_2norm}  {error2_2norm}  {diff_2norm}");

    f.write("\n");
