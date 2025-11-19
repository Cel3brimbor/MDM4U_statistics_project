import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

X = np.array([64, 128, 256, 512, 1024])
Y = np.array([0.00423, 0.00233, 0.00091, 0.0003, 0.00006])

A0_fit = 0.00710611
k_fit = 0.00830908

def exponential_decay(x, A0, k):
    return A0 * np.exp(-k * x)

X_fit = np.linspace(X.min(), X.max(), 500)

Y_fit = exponential_decay(X_fit, A0_fit, k_fit)

plt.figure(figsize=(10, 6))

plt.scatter(X, Y, label='Original Data', color='blue', marker='o', s=100)

plt.plot(X_fit, Y_fit, label=f'Fit: y = {A0_fit:.4f}$e^{{-{k_fit:.4f}x}}$', color='red', linestyle='--')

plt.title('Exponential Decay Fit: Maximum lambda Before Underfitting vs. Neurons per Layer', fontsize=14)
plt.xlabel('Neurons per Layer', fontsize=12)
plt.ylabel('Maximum lambda Before Underfitting', fontsize=12)

plt.legend(fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

plt.savefig('exponential_decay_fit.png')