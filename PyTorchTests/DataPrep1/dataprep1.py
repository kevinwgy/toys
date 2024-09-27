# MNIST Fashion data

import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import random

# Define training and testing data sets
training_data = datasets.FashionMNIST(
  root="data",
  train=True,
  download=True, #downloads data only if it is not in ``root''
  transform=ToTensor()
)
testing_data = datasets.FashionMNIST(
  root="data",
  train=False,
  download=True, #downloads data only if it is not in ``root''
  transform=ToTensor()
)

# Map labels to words 
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

Ntrain = len(training_data);
Ntest  = len(testing_data);
print("Size of traing data: ", Ntrain, ", Size of testing data", Ntest, ".");

# Plot 9 random images
figure = plt.figure(figsize=(8, 8));
cols = 3;
rows = 3;
for i in range(1, cols*rows + 1):
    index = random.randint(0, Ntrain);
    img, label = training_data[index];
    figure.add_subplot(rows, cols, i);
    plt.title(labels_map[label]);
    plt.axis("off");
    plt.imshow(img.squeeze(), cmap="gray");
plt.show();







