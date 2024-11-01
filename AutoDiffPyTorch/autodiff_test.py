import torch;

def f(x):
  return x**2;

def g(x):
  if x==1.0:
    return x*1.0; 
  else:
    return x**2;

x = torch.ones(1, requires_grad=True);
fx = f(x);
x2 = torch.ones(1, requires_grad=True);
gx = g(x2);

#print(f"Gradient of fun = {z.grad_fn}");

fx.backward();
gx.backward();

print(f"f'({x.detach().numpy()}) = {x.grad.numpy()}");
print(f"g'({x2.detach().numpy()}) = {x2.grad.numpy()}");
