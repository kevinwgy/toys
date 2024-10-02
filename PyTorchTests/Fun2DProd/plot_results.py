import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

#--------------------------
# Load model
#--------------------------
model = torch.jit.load('model_scripted.pt')
model.eval();

# ---------------------------------------------
# Load training data and validation/test data
# ---------------------------------------------
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, filename):
    self.filename = filename;
    rawdata = np.genfromtxt(self.filename);
    ncol = rawdata.shape[1];
    self.X = rawdata[:,0:ncol-2];
    self.F = rawdata[:,ncol-2:ncol-1];
    self.Label = rawdata[:,ncol-1:ncol];
  def __len__(self):
    return len(self.F);
  def __getitem__(self, idx):
    return self.X[idx,:], self.F[idx];
  def plot(self):
    return plt.scatter(self.X[:,0], self.X[:,1]);

training_data   = MyDataset("training_data.txt");
validation_data = MyDataset("validation_data.txt");

# ---------------------------------------------
# Plot
# ---------------------------------------------
x = np.linspace(-40.0, 40.0, 200);
y = np.linspace(-40.0, 40.0, 200);
z = np.array([i*j for j in y for i in x]);
X, Y = np.meshgrid(x,y);
Z = z.reshape(200,200);
plt.figure(figsize=(12,10));
plt.pcolormesh(X,Y,Z,vmin=-1600.0,vmax=1600.0,cmap='jet',shading='auto');
plt.colorbar();
tplot = training_data.plot();
tplot.set_facecolor('none');
tplot.set_edgecolor('r');
tplot.set_linewidth(2);
vplot = validation_data.plot();
vplot.set_facecolor('none');
vplot.set_edgecolor('b');
vplot.set_linewidth(2);
plt.savefig('ref2.png')
plt.draw();

xy = torch.from_numpy(np.array([[i,j] for j in y for i in x]));
xy = xy.float();
z = model(xy).detach().numpy();
X, Y = np.meshgrid(x,y);
Z = z.reshape(200,200);
plt.figure(figsize=(12,10));
plt.pcolormesh(X,Y,Z,vmin=-1600.0,vmax=1600.0,cmap='jet',shading='auto');
plt.colorbar();
plt.savefig('extrap.png')
plt.draw();

plt.show();
