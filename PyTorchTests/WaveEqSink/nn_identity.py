import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import copy


###########################################
# USER'S INPUTS
###########################################
model_type = "neural_network";
step_size = 0.001;

objective_fun = nn.MSELoss(); #least squares
optimizer_type = "Adam";
data_batch_size = 25;
Niter = 1;
###########################################


# ----------------------------------------------------------------------------------
# A neural network based on linear operators & common activation fuctions
# ----------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Identity()
  def forward(self, x):
    return self.layers(x)


# ----------------------------------------------------------------------------------
# 1. Choose model
# ----------------------------------------------------------------------------------
model = NeuralNetwork() if model_type=="neural_network" else NeuralNetwork();

device = "cpu"; #("cuda" if torch.cuda.is_available() else "cpu");
print(f"Using {device} device");

model = model.to(device); #run it on "device"
print(model);
    

# ----------------------------------------------------------------------------------
# 2. Load training data and validation/test data from file
# ----------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, filename):
    self.filename = filename;
    rawdata = np.genfromtxt(self.filename);
    ncol = rawdata.shape[1];
    self.X = rawdata[:, 2:5];
    self.F = rawdata[:,ncol-7:ncol-1];
# Currently the 'label' is not being used 
    self.Label = rawdata[:,ncol-1:ncol];  
  def __len__(self):
    return len(self.F);
  def __getitem__(self, idx):
    return self.X[idx,:], self.F[idx], self.Label[idx,:];
  def plot(self):
    return plt.scatter(self.X[:,0], self.X[:,1]);
    

training_data   = MyDataset("training_data.txt");
validation_data = MyDataset("validation_data.txt");


# ----------------------------------------------------------------------------------
# 3. Setup training method & parameters 
# ----------------------------------------------------------------------------------
#if optimizer_type == "SGD":
#  optimizer = torch.optim.SGD(model.parameters(), lr=step_size); #stochastic gradient descent
#elif optimizer_type == "Adam":
#  optimizer = torch.optim.Adam(model.parameters(), lr=step_size); #stochastic gradient descent

# setup the data loaders
training_dataloader   = torch.utils.data.DataLoader(training_data,
                                                    batch_size=data_batch_size, shuffle=True);
validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                    batch_size=data_batch_size, shuffle=True);


      
# ----------------------------------------------------------------------------------
# 4. Idenitity validation
# ----------------------------------------------------------------------------------

size = len(validation_dataloader.dataset);
model.eval(); #turn on evaluation mode 
num_batches = len(validation_dataloader);
with torch.no_grad(): #do not calculate gradients
  with open("scaled_debug.txt", "w") as f:
    for batch, (X, Y0, E0) in enumerate(validation_dataloader):
      Y  = model(Y0.float());
 
      size_types = Y.size(dim=1) // 2
      # scale predictions for conservation
      E = torch.sum(Y[:,0:size_types], dim=1) - torch.sum(Y[:,size_types:], dim=1)
      E = E.unsqueeze(1)
      YScale = torch.div(torch.mul(Y, E0), E)

      # write to a file
      f.write(f"## Batch {batch}\n")
      num_items = Y0.size(dim=0)
      for i in range(num_items):
        for v0, v in zip(Y0[i, :], YScale[i, :]):
          f.write(f"{v0:16.8e}{v:16.8e}")
        f.write("\n")
