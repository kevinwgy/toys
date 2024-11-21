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
Niter = 500; #can be increased to 1000 for slightly better prediction
###########################################


# ----------------------------------------------------------------------------------
# A neural network based on linear operators & common activation fuctions
# ----------------------------------------------------------------------------------
class NeuralNetwork(nn.Module):
  def __init__(self, size_in : int, size_out : int):
    super().__init__()
    self.size_in = size_in
    self.size_out = size_out
    self.layers = nn.Sequential(
      nn.Linear(size_in, 20),  
      nn.ReLU(),
      nn.Linear(20, 20),
      nn.ReLU(),
      nn.Linear(20, 20),
      nn.ReLU(),
      nn.Linear(20, size_out),
    )


  def forward(self, X):

    num_items = self.size_out // 2; # divide by 2 and drop the remainder
    x = X[0:self.size_in] if X.ndim==1 else X[:,0:self.size_in]; #get col vector


    # non-dimensionalization
    tc = 1.0;  


    out = self.layers(x)

    e0 = X[0] if X.ndim==1 else X[:,0];  #true conserved quantity
    
    if X.ndim==1:
      e   = torch.sum(out[:num_items]) - torch.sum(out[num_items:])
    else:
      e   = torch.sum(out[:, :num_items], dim=1) - torch.sum(out[:, num_items:], dim=1)
      e   = e.unsqueeze(1)
      e0  = e0.unsqueeze(1)
    
    return torch.div(torch.mul(out, e0), e)


# ----------------------------------------------------------------------------------
# 1. Choose model
# ----------------------------------------------------------------------------------
model = NeuralNetwork(3,6) if model_type=="neural_network" else NeuralNetwork(3,6); #add choices later

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
    self.X = rawdata[:, 0:3];  #rows 0, 1, 2 are inputs
    self.F = rawdata[:,ncol-6:ncol]; #six outputs
  def __len__(self):
    return len(self.F);
  def __getitem__(self, idx):
    return self.X[idx,:], self.F[idx];
  #def plot(self):
  #  return plt.scatter(self.X[:,0], self.X[:,1]);
    

training_data   = MyDataset("DataPrep/training_data.txt");
validation_data = MyDataset("DataPrep/validation_data.txt");


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
  size = len(dataloader.dataset);
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

# ----------------------------------------------------------------------------------
# 5. Plotting the iteration history
# ----------------------------------------------------------------------------------
# Hold the best model

plt.figure(figsize=(10, 6))
plt.semilogy(history)  # Use semilogy for logarithmic scale on the y-axis
plt.title(' Average Error VS Iterations')
plt.xlabel('Iterations')
plt.ylabel('Error')
plt.grid(True)  # Optional: Add grid for better visibility
plt.savefig('hist.png')  # Save the plot
plt.draw()  # Update the plot
