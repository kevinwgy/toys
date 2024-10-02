import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import copy


###########################################
# USER'S INPUTS
###########################################
#model_type = "polynomial";
#polynomial_order = 3;
#step_size = 0.1;

model_type = "neural_network";
step_size = 0.01;

objective_fun = nn.MSELoss(); #least squares
optimizer_type = "Adam";
data_batch_size = 50;
Niter = 500;
###########################################


# ----------------------------------------------------------------------------------
# A custom 2D polynomial regression ``layer'' (equivalently, a single-layer network)
#
# f(x,y) = (a0+a1*x+a2*x^2+...)*(b0+b1*y+b2*x^2+...)
# ----------------------------------------------------------------------------------
class PolyRegression2D(nn.Module):
  def __init__(self, size_in, size_out): #size_in = 2, size_out = 1 (REQUIRED)
    super().__init__();
    self.size_in  = size_in;
    self.size_out = size_out;
    self.order = polynomial_order; #order of polynomial, hard-coded
    #define parameters (automatically setting requires_grad=True)
    self.a = nn.Parameter(torch.ones(self.order+1));
    self.b = nn.Parameter(torch.ones(self.order+1));
  def forward(self, X): 
    N = 1 if X.ndim==1 else X.size(dim=0);
    x  = X[0]*torch.ones(1) if X.ndim==1 else X[:,0:1]; #get col vector
    y  = X[1]*torch.ones(1) if X.ndim==1 else X[:,1:2];
    xp = torch.ones_like(x);  #x^0, x^1, x^2, ...
    yp = torch.ones_like(y);  #y^0, y^1, y^2, ...
    fx = self.a[0]*torch.ones_like(x);
    fy = self.b[0]*torch.ones_like(y);
    for i in range(1,self.order+1):
      xp = xp*x;
      yp = yp*y; 
      fx = fx + self.a[i]*xp;
      fy = fy + self.b[i]*yp;
    return fx*fy;

# ----------------------------------------------------------------------------------
# A neural network based on linear operators & common activation fuctions
# ----------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(2, 10),
      nn.ReLU(),
      nn.Linear(10, 10),
      nn.ReLU(),
      nn.Linear(10, 10),
      nn.ReLU(),
      nn.Linear(10, 10),
      nn.ReLU(),
      nn.Linear(10, 1),
    )
  def forward(self, x):
    return self.layers(x)


# ----------------------------------------------------------------------------------
# 1. Choose model
# ----------------------------------------------------------------------------------
model = PolyRegression2D(2,1) if model_type=="polynomial" else NeuralNetwork();

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


# ----------------------------------------------------------------------------------
# 3. Setup training method & parameters 
# ----------------------------------------------------------------------------------
if optimizer_type == "SGD":
  optimizer = torch.optim.SGD(model.parameters(), lr=step_size); #stochastic gradient descent
elif optimizer_type == "Adam":
  optimizer = torch.optim.Adam(model.parameters(), lr=step_size); #stochastic gradient descent

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
    Y   = model(X.float());  #"Linear" expects "float", unable to handle "double"...
    obj = obj_fn(Y, Y0.float());
    optimizer.zero_grad(); #reset gradients of model parameters to zero.
    obj.backward();
    optimizer.step();

    if batch % 20 == 0: #print info to screen
      obj     = obj.item();
      current = batch*data_batch_size + len(X);
      print(f"loss: {obj:>7f}  [{current:>5d}/{size:>5d}]");

def validation(dataloader, model, obj_fn):
  model.eval(); #turn on evaluation mode 
  num_batches = len(dataloader);
  test_obj = 0.0;
  with torch.no_grad(): #do not calculate gradients
    for X, Y0 in dataloader:
      Y  = model(X.float());
      test_obj += float(obj_fn(Y, Y0.float()).item());
  test_obj /= num_batches;
  print(f"Avg error: {test_obj:>8f} \n")
  return test_obj;
      

# ----------------------------------------------------------------------------------
# 4. Training the model
# ----------------------------------------------------------------------------------
# Hold the best model
best_obj = np.inf;
best_params = None;
history = [];

for it in range(Niter):
  print(f"Iteration {it}:\n");
  training(training_dataloader, model, objective_fun, optimizer);
  test_obj = validation(validation_dataloader, model, objective_fun);
  history.append(test_obj);
  if test_obj < best_obj:
    best_obj = test_obj;
    best_params = copy.deepcopy(model.state_dict());

# restore model that offers best accuracy
model.load_state_dict(best_params)

# save model to file
model_scripted = torch.jit.script(model) # Export to TorchScript
model_scripted.save('model_scripted.pt') # Save

print("Done!");
print(f"Best average error: {best_obj:>8f} \n");
plt.plot(history);
plt.savefig('hist.png');
plt.draw();


# ----------------------------------------------------------------------------------
# 5. Visualization
# ----------------------------------------------------------------------------------
if True:
  x = np.linspace(0, 1.0, 200);
  y = np.linspace(0, 1.0, 200);
  z = np.array([np.cos(2.0*np.pi*i)*np.sin(2.0*np.pi*j) for j in y for i in x]);
  X, Y = np.meshgrid(x,y);
  Z = z.reshape(200,200);
  plt.figure(figsize=(12,10));
  plt.pcolormesh(X,Y,Z,vmin=-1,vmax=1,shading='auto');
  plt.colorbar();
  tplot = training_data.plot();
  tplot.set_facecolor('none');
  tplot.set_edgecolor('r');
  tplot.set_linewidth(2);
  vplot = validation_data.plot();
  vplot.set_facecolor('none');
  vplot.set_edgecolor('b');
  vplot.set_linewidth(2);
  plt.savefig('ref1.png');
  plt.draw()


if True:
  x = np.linspace(0, 1, 200);
  y = np.linspace(0, 1, 200);
  xy = torch.from_numpy(np.array([[i,j] for j in y for i in x]));
  xy = xy.float();
  z = model(xy).detach().numpy();
  X, Y = np.meshgrid(x,y);
  Z = z.reshape(200,200);
  plt.figure(figsize=(12,10));
  plt.pcolormesh(X,Y,Z,vmin=-1,vmax=1,shading='auto');
  plt.colorbar();
  plt.savefig('interp.png');
  plt.draw()

plt.show();

