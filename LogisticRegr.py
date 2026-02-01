import sys
import math
import numpy as np
import matplotlib.pyplot as plt

data = np.array([
    [34, 78, 0],
    [45, 85, 0],
    [50, 43, 1],
    [65, 72, 1],
    [70, 90, 1]
], dtype=float)

X = data[:, :2]
y = data[:, 2].astype(int)

X_mean = X.mean(axis=0)
X_std = X.std(axis=0, ddof=0)
# protect against zero std
X_std = np.where(X_std == 0, 1.0, X_std)
Xs = (X - X_mean) / X_std

# sklearn LogisticRegression
use_sklearn = True
try:
    from sklearn.linear_model import LogisticRegression
except Exception:
    use_sklearn = False

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

if use_sklearn:
    model = LogisticRegression(solver='liblinear')
    model.fit(Xs, y)
    intercept = float(model.intercept_[0])
    coef = model.coef_.ravel().astype(float)
    def predict_proba_matrix(Xraw):
        Xs_local = (np.asarray(Xraw, dtype=float) - X_mean) / X_std
        return model.predict_proba(Xs_local)[:, 1]
    def predict_class(Xraw):
        return (predict_proba_matrix(Xraw) >= 0.5).astype(int)
else:
    # Design matrix with intercept
    Xd = np.hstack([np.ones((Xs.shape[0], 1)), Xs])  # shape (N, 3)
    w = np.zeros(Xd.shape[1], dtype=float)          # [w0, w1, w2]
    lr = 0.8
    max_iter = 20000
    tol = 1e-8
    for it in range(max_iter):
        z = Xd.dot(w)
        p = sigmoid(z)
        grad = Xd.T.dot(p - y) / y.size
        w -= lr * grad
        if np.linalg.norm(grad) < tol:
            break
    intercept = float(w[0])
    coef = w[1:].astype(float)
    def predict_proba_matrix(Xraw):
        Xs_local = (np.asarray(Xraw, dtype=float) - X_mean) / X_std
        Xd_local = np.hstack([np.ones((Xs_local.shape[0], 1)), Xs_local])
        return sigmoid(Xd_local.dot(w))
    def predict_class(Xraw):
        return (predict_proba_matrix(Xraw) >= 0.5).astype(int)


probs = predict_proba_matrix(X)
preds = predict_class(X)

print("Fitted logistic regression model (on standardized features):")
print(f"Intercept: {intercept:.6f}")
print(f"Coefficients (for standardized Exam1, Exam2): {coef[0]:.6f}, {coef[1]:.6f}\n")

print("Training samples (Exam1, Exam2) -> prob(class=1), true, pred")
for i, (xi, yi) in enumerate(zip(X, y)):
    print(f"Sample {i+1}: ({int(xi[0])}, {int(xi[1])}) -> prob={probs[i]:.4f}, true={yi}, pred={int(preds[i])}")

accuracy = (preds == y).mean()
tp = int(np.sum((preds==1) & (y==1)))
tn = int(np.sum((preds==0) & (y==0)))
fp = int(np.sum((preds==1) & (y==0)))
fn = int(np.sum((preds==0) & (y==1)))

print(f"\nTraining accuracy: {accuracy*100:.2f}%")
print(f"Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}\n")

print("Decision function (original units):")
print(f"logit(x) = {intercept:.6f} + {coef[0]:.6f}*(Exam1 - {X_mean[0]:.3f})/{X_std[0]:.3f}"
      f" + {coef[1]:.6f}*(Exam2 - {X_mean[1]:.3f})/{X_std[1]:.3f}")
print("Probability = sigmoid(logit)\n")

pad = 6
x1_min, x1_max = X[:,0].min()-pad, X[:,0].max()+pad
x2_min, x2_max = X[:,1].min()-pad, X[:,1].max()+pad
g1 = np.linspace(x1_min, x1_max, 300)
g2 = np.linspace(x2_min, x2_max, 300)
G1, G2 = np.meshgrid(g1, g2)
grid = np.c_[G1.ravel(), G2.ravel()]
grid_probs = predict_proba_matrix(grid).reshape(G1.shape)

plt.figure(figsize=(8,7))
# p=0.5 contour
contour = plt.contour(G1, G2, grid_probs, levels=[0.5], colors='purple', linewidths=2)
# scatter points
markers = {0: 'o', 1: 's'}
colors = {0: 'orange', 1: 'tab:red'}
for label in np.unique(y):
    idx = (y == label)
    plt.scatter(X[idx,0], X[idx,1], marker=markers[label], s=100,
                edgecolors='k', label=f"Admitted={label}", color=colors[label])
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.title("Logistic Regression Decision Boundary (p=0.5)")
plt.legend()
plt.grid(alpha=0.4)
plt.tight_layout()
plt.savefig("decision_boundary.png", dpi=150)
print("Saved decision boundary plot to decision_boundary.png (also displayed).")

plt.show()

# ExAmple
new_student = np.array([[60, 75]])
p_new = predict_proba_matrix(new_student)[0]
print(f"Example: New student (Exam1=60, Exam2=75) -> prob of admission = {p_new:.4f}, class = {int(p_new>=0.5)}")
