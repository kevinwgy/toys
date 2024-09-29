import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


###########################################
# USER'S INPUTS
###########################################
polynomial_order = 3;
# define the training method & its parameters 
step_size = 0.6;
objective_fun = nn.MSELoss(); #least squares
optimizer_type = "SGD";
data_batch_size = 50;
Niter = 1500;
###########################################


# ----------------------------------------------------------------------------------
# A custom 1D polynomial regression ``layer'' (equivalently, a single-layer network)
#
# f(x) = a0+a1*x+a2*x^2+a3*x^3+...
# ----------------------------------------------------------------------------------
class PolyRegression1D(nn.Module):
  def __init__(self, size_in, size_out): #size_in = 1, size_out = 1 (REQUIRED)
    super().__init__();
    self.size_in  = size_in;
    self.size_out = size_out;
    self.order = polynomial_order; #order of polynomial, hard-coded
    #define parameters (automatically setting requires_grad=True)
    self.a = nn.Parameter(torch.zeros(self.order+1));
  def forward(self, X): 
    N = 1 if X.ndim==1 else X.size(dim=0);
    x  = X[0]*torch.ones(1) if X.ndim==1 else X[:,0:1]; #get col vector
    xp = torch.ones_like(x);  #x^0, x^1, x^2, ...
    fx = self.a[0]*torch.ones_like(x);
    for i in range(1,self.order+1):
      xp = xp*x;
      fx = fx + self.a[i]*xp;
    return fx;

# ----------------------------------------------------------------------------------
# Choose model
# ----------------------------------------------------------------------------------
model = PolyRegression1D(1,1);  #input dim = 1, output dim = 1

device = "cpu"; #("cuda" if torch.cuda.is_available() else "cpu");
print(f"Using {device} device");

model = model.to(device); #run it on "device"
print(model);
    

# ----------------------------------------------------------------------------------
# Load training data and validation/test data from file
# ----------------------------------------------------------------------------------
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
    return plt.scatter(self.X[:,0], self.F[:,0]);
    

training_data   = MyDataset("training_data.txt");
validation_data = MyDataset("validation_data.txt");


# ----------------------------------------------------------------------------------
# Setup training method & parameters 
# ----------------------------------------------------------------------------------
assert(optimizer_type == "SGD");
optimizer = torch.optim.SGD(model.parameters(), lr=step_size); #stochastic gradient descent

# setup the data loaders
training_dataloader   = torch.utils.data.DataLoader(training_data,
                                                    batch_size=data_batch_size, shuffle=True);
validation_dataloader = torch.utils.data.DataLoader(validation_data,
                                                    batch_size=data_batch_size, shuffle=True);


# define training and validation functions for each iteration of optimization
def training(dataloader, model, obj_fn, optimizer):
  size = len(dataloader.dataset);
  model.train(); #turn on training mode
  for batch, (X, Y0) in enumerate(dataloader):
    Y   = model(X);
    obj = obj_fn(Y, Y0);

    obj.backward();
    optimizer.step();
    optimizer.zero_grad(); #reset gradients of model parameters to zero.

    if batch % 100 == 0: #print info to screen
      obj     = obj.item();
      current = batch*data_batch_size + len(X);
      print(f"loss: {obj:>7f}  [{current:>5d}/{size:>5d}]");

def validation(dataloader, model, obj_fn):
  size = len(dataloader.dataset);
  model.eval(); #turn on evaluation mode 
  num_batches = len(dataloader);
  test_obj = 0.0;
  with torch.no_grad(): #do not calculate gradients
    for X, Y0 in dataloader:
      Y  = model(X);
      test_obj += obj_fn(Y, Y0).item();
  test_obj /= num_batches;
  print(f"Avg error: {test_obj:>8f} \n")
      

# training...
for t in range(Niter):
  print(f"It. {t+1}\n");
  training(training_dataloader, model, objective_fun, optimizer);
  validation(validation_dataloader, model, objective_fun);
print("Done!");


#visualization
if True:
  plt.figure(figsize=(12,10));

  # training and validation data points
  tplot = training_data.plot();
  tplot.set_facecolor('none');
  tplot.set_edgecolor('r');
  tplot.set_linewidth(1);
  vplot = validation_data.plot();
  vplot.set_facecolor('none');
  vplot.set_edgecolor('b');
  vplot.set_linewidth(1);

  # model prediction
  x = np.linspace(0, 1.0, 200);
  xtensor = torch.from_numpy(x).reshape(-1,1); #"-1" means the number will be found automatically
  z = model(xtensor).detach().numpy().reshape(-1); #convert to a 1-d array
  plt.plot(x,z,linewidth=3,color='green');

  plt.draw()

plt.show();

