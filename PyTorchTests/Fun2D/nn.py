import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


device = "cpu"; #("cuda" if torch.cuda.is_available() else "cpu");
print(f"Using {device} device");

# ----------------------------------------------------------------------------------
# A custom 2D polynomial regression ``layer'' (equivalently, a single-layer network)
#
# f(x,y) = (a0+a1*x+a2*x^2+a3*x^3)*(b0+b1*y+b2*x^2+b3*x^3)
# ----------------------------------------------------------------------------------
class PolyRegression2D(nn.Module):
  def __init__(self, size_in, size_out): #size_in = 2, size_out = 1 (REQUIRED)
    super().__init__();
    self.size_in  = size_in;
    self.size_out = size_out;
    self.order = 4; #order of polynomial, hard-coded
    #define parameters (automatically setting requires_grad=True)
    self.a = nn.Parameter(torch.zeros(self.order+1));
    self.b = nn.Parameter(torch.zeros(self.order+1));
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
      nn.Linear(2, 4),
      nn.ReLU(),
      nn.Linear(4, 4),
      nn.ReLU(),
      nn.Linear(4, 1),
    )
  def forward(self, x):
    return self.layers(x)


# ----------------------------------------------------------------------------------
# Choose model
# ----------------------------------------------------------------------------------
#model = PolyRegression2D(2,1);  #input dim = 2, output dim = 1
model = NeuralNetwork();
model = model.to(device); #run it on "device"
print(model);
    

# ----------------------------------------------------------------------------------
# Load training data and validation/test data from file
# ----------------------------------------------------------------------------------
class MyDataset(torch.utils.data.Dataset):
  def __init__(self, filename):
    self.filename = filename;
    rawdata = np.genfromtxt(self.filename);
    self.X = rawdata[:,0:2];
    self.F = rawdata[:,2:3];
    self.Label = rawdata[:,3:4];  
  def __len__(self):
    return len(self.F);
  def __getitem__(self, idx):
    return self.X[idx,:], self.F[idx];

training_data   = MyDataset("training_data.txt");
validation_data = MyDataset("validation_data.txt");

#visualization
x = np.linspace(0, 20*np.pi, 100);
y = np.linspace(0, 20*np.pi, 100);
z = np.array([np.cos(i)*np.sin(j) for j in y for i in x]);
X, Y = np.meshgrid(x,y);
Z = z.reshape(100,100);
plt.contourf(X,Y,Z,100);
plt.show()


#debug_dataloader = torch.utils.data.DataLoader(training_data, batch_size=64, shuffle=True);
#train_input, train_output = next(iter(debug_dataloader));
#print(train_input);
#print(train_output);

# ----------------------------------------------------------------------------------
# Training the model 
# ----------------------------------------------------------------------------------
# define the training method & its parameters 
step_size = 0.001;
objective_fun = nn.MSELoss; #least squares
optimizer = torch.optim.SGD(model.parameters(), lr=step_size); #stochastic gradient descent
data_batch_size = 64;
Niter = 10;

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
      current = batch*batch_size + len(X);
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
  print(f"Avg error: {test_loss:>8f} \n")
      
# training...
for t in range(Niter):
  print(f"It. {t+1}\n");
  training(training_dataloader, model, objective_fun, optimizer);
  validation(validation_dataloader, model, objective_fun);
print("Done!");


X = 2.0*torch.ones(3,2); #one input, 2 dimensions (x,y)
Y = model(X);
Y = Y.to('cpu');
#print(Y.detach().numpy());
print(f"Prediction: y({X[0,0]},{X[0,1]})={Y.detach().numpy()[0,0]}")

