import numpy as np

def stats_xy(X, Y, ddof=0, tol=1e-12):
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape:
        print("X and Y must have same length.")
    N = X.size
    if N == 0:
        print("Empty data.")
    mean_x = X.mean()
    mean_y = Y.mean()
    var_x = ((X - mean_x) ** 2).sum() / (N - ddof)
    var_y = ((Y - mean_y) ** 2).sum() / (N - ddof)
    cov = ((X - mean_x) * (Y - mean_y)).sum() / (N - ddof)
    if var_x <= 0 or var_y <= 0:
        corr = np.nan
    else:
        corr = cov / (math.sqrt(var_x) * math.sqrt(var_y))
    # Interpret correlation/covariance
    if math.isclose(cov, 0.0, abs_tol=tol):
        cov_msg = "No linear relationship between the two features (covariance ≈ 0)."
    elif cov > 0:
        cov_msg = "Positive covariance."
    else:
        cov_msg = "Negative covariance."

    if math.isclose(corr, 0.0, abs_tol=tol):
        corr_msg = "Correlation ≈ 0 (no linear correlation)."
    elif corr > 0:
        corr_msg = "Positive correlation."
    else:
        corr_msg = "Negative correlation."

    return {
        "mean_x": mean_x,
        "mean_y": mean_y,
        "var_x": var_x,
        "var_y": var_y,
        "covariance": cov,
        "correlation": corr,
        "cov_msg": cov_msg,
        "corr_msg": corr_msg
    }

# Example
import math
X = [2.0, 4.0, 6.0, 8.0]
Y = [1.0, 3.0, 5.0, 7.0]
res = stats_xy(X, Y, ddof=0)
for k, v in res.items():
    print(f"{k}: {v}")
