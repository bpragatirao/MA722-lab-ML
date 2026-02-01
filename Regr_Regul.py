import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0

def add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])

def center_data(X, y):
    X_mean = X.mean(axis=0)
    y_mean = y.mean()
    Xc = X - X_mean
    yc = y - y_mean
    return Xc, yc, X_mean, y_mean

# Linear Regression (normal eqn) on centered X (no intercept regularized)
def linear_regr(X, y):
    Xc, yc, X_mean, y_mean = center_data(X, y)
    # Beta on centered data (no intercept)
    # beta = (Xc^T Xc)^{-1} Xc^T yc
    XtX = Xc.T @ Xc
    try:
        beta = np.linalg.solve(XtX, Xc.T @ yc)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(XtX) @ (Xc.T @ yc)
    intercept = y_mean - X_mean @ beta
    return intercept, beta

# Ridge Regression closed form
def ridge_closed(X, y, lam=1.0):
    Xc, yc, X_mean, y_mean = center_data(X, y)
    n_features = Xc.shape[1]
    XtX = Xc.T @ Xc
    A = XtX + lam * np.eye(n_features)
    try:
        beta = np.linalg.solve(A, Xc.T @ yc)
    except np.linalg.LinAlgError:
        beta = np.linalg.pinv(A) @ (Xc.T @ yc)
    intercept = y_mean - X_mean @ beta
    return intercept, beta

# Lasso via Coordinate Descent
def soft_threshold(rho, lam):
    if rho > 0 and lam < abs(rho):
        return rho - lam
    if rho < 0 and lam < abs(rho):
        return rho + lam
    return 0.0

def lasso_coord(X, y, lam=0.1, max_iter=10000, tol=1e-6, standardize=False):
    n_samples, n_features = X.shape
    Xc, yc, X_mean, y_mean = center_data(X, y)
    X_work = Xc.copy()
    if standardize:
        norms = np.sqrt((X_work ** 2).sum(axis=0) / n_samples)
        norms = np.where(norms == 0, 1.0, norms)
        X_work = X_work / norms
    else:
        norms = np.ones(n_features)

    # initialize
    beta = np.zeros(n_features, dtype=float)
    # precompute column squared norms (1/n) * sum x_j^2
    X_col_sq = (X_work ** 2).sum(axis=0) / n_samples

    for it in range(max_iter):
        beta_max_change = 0.0
        for j in range(n_features):
            # compute rho_j = (1/n) * sum_i x_ij * (y_i - sum_{k != j} x_ik b_k)
            # Efficient: residual + x_j * beta_j contains (y - X b) + x_j * b_j
            residual = yc - X_work @ beta
            rho = (X_work[:, j] * (residual + X_work[:, j] * beta[j])).sum() / n_samples
            # update with soft-threshold
            new_beta_j = soft_threshold(rho, lam) / (X_col_sq[j] + 1e-16)
            change = abs(new_beta_j - beta[j])
            if change > beta_max_change:
                beta_max_change = change
            beta[j] = new_beta_j
        if beta_max_change < tol:
            break

    # convert back if standardized
    beta_unscaled = beta / norms
    intercept = y_mean - X_mean @ beta_unscaled
    return intercept, beta_unscaled


def fit_and_report(X, y, lasso_lambda=0.1, ridge_lambda=1.0, show_plots=True):
    # Fit models
    intercept_lin, beta_lin = linear_regr(X, y)
    intercept_ridge, beta_ridge = ridge_closed(X, y, lam=ridge_lambda)
    intercept_lasso, beta_lasso = lasso_coord(X, y, lam=lasso_lambda, standardize=True)

    # Predictions
    def predict(intercept, beta, X_):
        return X_ @ beta + intercept

    y_lin = predict(intercept_lin, beta_lin, X)
    y_ridge = predict(intercept_ridge, beta_ridge, X)
    y_lasso = predict(intercept_lasso, beta_lasso, X)

    # Metrics
    metrics = {
        'Linear': (mse(y, y_lin), r2_score(y, y_lin)),
        'Ridge':  (mse(y, y_ridge), r2_score(y, y_ridge)),
        'Lasso':  (mse(y, y_lasso), r2_score(y, y_lasso))
    }

    # Print coefficients
    print("\nModel coefficients (intercept first):")
    print(f"Linear  : intercept={intercept_lin:.6f}, beta={np.round(beta_lin,6)}")
    print(f"Ridge   : intercept={intercept_ridge:.6f}, beta={np.round(beta_ridge,6)} (lam={ridge_lambda})")
    print(f"Lasso   : intercept={intercept_lasso:.6f}, beta={np.round(beta_lasso,6)} (lam={lasso_lambda})")

    print("\nMetrics (MSE, R2):")
    for m, (mi, ri) in metrics.items():
        print(f"{m:6s} -> MSE: {mi:.6f}, R2: {ri:.6f}")

    # Plotting
    n_features = X.shape[1]
    if show_plots:
        if n_features == 1:
            plt.figure(figsize=(8,6))
            plt.scatter(X[:,0], y, label='Data', s=60, edgecolor='k')
            x_line = np.linspace(X[:,0].min(), X[:,0].max(), 200)
            plt.plot(x_line, intercept_lin + beta_lin[0]*x_line, label='Linear', linewidth=2)
            plt.plot(x_line, intercept_ridge + beta_ridge[0]*x_line, label=f'Ridge (λ={ridge_lambda})', linewidth=2)
            plt.plot(x_line, intercept_lasso + beta_lasso[0]*x_line, label=f'Lasso (λ={lasso_lambda})', linewidth=2)
            plt.legend()
            plt.xlabel("Feature 1")
            plt.ylabel("Target")
            plt.title("Regression Lines (Single Feature)")
            plt.grid(alpha=0.3)
            plt.show()
        else:
            # Bar chart of coefficients
            indices = np.arange(n_features)
            width = 0.25
            plt.figure(figsize=(10,6))
            plt.bar(indices - width, beta_lin, width=width, label='Linear')
            plt.bar(indices, beta_ridge, width=width, label=f'Ridge (λ={ridge_lambda})')
            plt.bar(indices + width, beta_lasso, width=width, label=f'Lasso (λ={lasso_lambda})')
            plt.xlabel("Feature index")
            plt.ylabel("Coefficient value")
            plt.title("Model Coefficients Comparison")
            plt.legend()
            plt.grid(alpha=0.3)
            plt.show()

            # Bar chart for MSE
            names = list(metrics.keys())
            mses = [metrics[n][0] for n in names]
            plt.figure(figsize=(6,4))
            plt.bar(names, mses, color=['C0','C1','C2'])
            plt.ylabel("MSE")
            plt.title("Model MSE Comparison")
            plt.grid(axis='y', alpha=0.3)
            plt.show()

    return (intercept_lin, beta_lin), (intercept_ridge, beta_ridge), (intercept_lasso, beta_lasso), metrics

print("Regression suite (Linear, Ridge, Lasso) - no scikit-learn for models.")
path = input("Enter CSV file path (features in columns, target in last column). Press Enter to run synthetic demo: ").strip()
if path == "":
    # simple synthetic demo (multi-feature)
    print("Running synthetic demo dataset (3 features)...")
    np.random.seed(0)
    n = 80
    X = np.random.randn(n, 3)
    true_beta = np.array([2.0, -1.0, 0.5])
    y = X @ true_beta + 0.5 * np.random.randn(n)
else:
    df = pd.read_csv(path)
    arr = df.values
    if arr.shape[1] < 2:
        print("CSV must have at least one feature and one target column.")
    X = arr[:, :-1].astype(float)
    y = arr[:, -1].astype(float)

print(f"Dataset shape: X={X.shape}, y={y.shape}")
# user-specified hyperparams
try:
    ridge_lambda = float(input("Enter Ridge λ (e.g., 1.0): ") or 1.0)
except:
    ridge_lambda = 1.0
try:
    lasso_lambda = float(input("Enter Lasso λ (e.g., 0.1): ") or 0.1)
except:
    lasso_lambda = 0.1

fit_and_report(X, y, lasso_lambda=lasso_lambda, ridge_lambda=ridge_lambda, show_plots=True)


