import numpy as np
import matplotlib.pyplot as plt

from os.path import dirname, abspath

filename = "conjecture1_alpha_1_5.txt"
path_ = dirname(abspath(__file__))
path = path_ + r"/Data/"
with open(path+filename, "r") as f:
    lines = f.readlines()

# Get N from the first line
N = int(lines[0].split()[2])

TE_peak_normalised = []
# Loop through the rest of the lines
for i in range(4, len(lines), 3):
    # Extract the network label
    network_label = int(lines[i].split()[0])
    
    # Extract the minimum of the first two values and the third value for the current network
    values = [float(lines[i+j].split()[2]) for j in range(2)]
    min_value = min(values)
    third_value = float(lines[i+2].split()[2])

    TE_peak_normalised.append(min_value/third_value)
    
    # Do whatever you want with the extracted values
    #print(f"Network {network_label}: minimum value = {min_value}, third value = {third_value}")

bin_size = 0.03

# compute histogram
bins = np.arange(0, 2, bin_size)
hist, _ = np.histogram(TE_peak_normalised, bins)

# print histogram
for i, count in enumerate(hist):
    bin_start = i * bin_size
    bin_end = bin_start + bin_size
    print(f"{bin_start:.1f}-{bin_end:.1f}: {count}")


# plot histogram
fig, ax = plt.subplots()
ax.hist(TE_peak_normalised, bins=bins, edgecolor='black')
ax.set_xlabel('$\beta$ of TE Peak Normalised')
ax.set_ylabel('Counts')
ax.set_title('Histogram of $\beta$ of TE Peak Normalised, N = 100')
ax.axvline(x=1.0, color='red', linestyle='--')

# set axis labels and title
plt.xlabel('beta of TE_peak_normalised')
plt.ylabel('Count')
plt.title(f'Histogram of TE_peak_normalised (bin size = {bin_size}), N = 500')

# show plot
plt.show()
