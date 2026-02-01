import numpy as np

def normal_mle(data):
    x = np.asarray(data, dtype=float)
    if x.size == 0:
        print("Empty data.")
    N = x.size
    mu_hat = x.mean()
    sigma2_hat = ((x - mu_hat)**2).sum() / N   # MLE
    return mu_hat, sigma2_hat

# Example
data = [1.2, 1.8, 0.5, 2.0, 1.7]
mu_hat, sigma2_hat = normal_mle(data)
print("MLE mean (mu):", mu_hat)
print("MLE variance (sigma^2):", sigma2_hat)
