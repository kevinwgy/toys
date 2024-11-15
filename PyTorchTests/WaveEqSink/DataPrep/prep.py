import numpy as np;
import subprocess;

rawdata = np.loadtxt('../DataGen/results.txt');

tru_conserved = rawdata[:,2]; #3rd column is the true conserved
num_conserved = rawdata[:,5] + rawdata[:,6] + rawdata[:,7] \
              - rawdata[:,8] - rawdata[:,9] - rawdata[:,10];

ratio = tru_conserved / num_conserved;
for i in range(5,11):
  rawdata[:,i] *= ratio;

# Write cleaned data to file
with open('results_clean.txt', "w") as out:
  for i in range(rawdata.shape[0]):
    for x in rawdata[i,2:11]: #ignore first two columns and last column
      out.write(f"{x:16.8e}");
    out.write("\n");

# Separate training and validation data
subprocess.run("./results2datagen.sh");
