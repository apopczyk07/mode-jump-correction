import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy import sparse
from scipy.sparse.linalg import spsolve

def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z

# Step 1: Read the signal from a file
file_path = 'Values.csv'  # Adjust this to your file path
data = pd.read_csv(file_path)

# Assuming the file has two columns, the second column is the signal
y = data.iloc[:, 1].values  # Adjust column index if necessary

# Ensure y is a 1D array
if len(y.shape) > 1:
    y = y.squeeze()

# Step 2: Apply ALS to estimate the baseline
baseline = baseline_als(y, lam=1e4, p=0.05)

# Step 3: Subtract the baseline from the original signal
corrected_signal = y - baseline

peaks, _ = find_peaks(corrected_signal, height=8)

# Step 4: Plot vertical lines at peak positions
plt.figure(figsize=(8, 6))
plt.vlines(x=peaks, ymin=0, ymax=5, color='red', linestyle='dashed')
plt.xlabel('Index')
plt.ylabel('Amplitude')
plt.title('Detected Peaks')
plt.grid(True)
plt.show()
