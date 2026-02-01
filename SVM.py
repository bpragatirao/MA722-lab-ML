import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

data = np.array([
    [10,  5, 1],
    [ 1,  1, 0],
    [ 5,  3, 0],
    [ 3,  2, 0],
    [15,  7, 1],
    [ 2,  1, 0],
    [ 7,  4, 0],
    [ 9,  6, 1],
    [20, 10, 1],
    [ 0,  0, 0]
], dtype=float)

X = data[:, :2]
y = data[:, 2].astype(int)

# ---- Train / Test split ----
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# ---- Train linear SVM ----
clf = SVC(kernel='linear', C=1.0)  # linear kernel
clf.fit(X_train, y_train)

# ---- Evaluate ----
y_train_pred = clf.predict(X_train)
y_test_pred  = clf.predict(X_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc  = accuracy_score(y_test, y_test_pred)

print("Training accuracy: {:.2f}%".format(train_acc * 100))
print("Test accuracy    : {:.2f}%".format(test_acc * 100))
print("\nConfusion matrix on test set:\n", confusion_matrix(y_test, y_test_pred))

# ---- Plot data, decision boundary and support vectors ----
plt.figure(figsize=(8,6))

# Scatter points colored by label
colors = ['tab:blue', 'tab:orange']
markers = ['o', 's']
for label in np.unique(y):
    idx = (y == label)
    plt.scatter(X[idx,0], X[idx,1], c=colors[label], marker=markers[label],
                label=f"spam={label}", edgecolors='k', s=90)

# Plot training points with a faint edge to indicate which were used for training
plt.scatter(X_train[:,0], X_train[:,1], facecolors='none', edgecolors='k', s=120, linewidths=1.0, alpha=0.5, label='train samples')

# Decision boundary from the linear SVM coefficients
w = clf.coef_[0]      # shape (2,)
b = clf.intercept_[0] # scalar

# Create grid to evaluate decision function
x_min, x_max = X[:,0].min() - 2, X[:,0].max() + 2
y_min, y_max = X[:,1].min() - 2, X[:,1].max() + 2
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = clf.decision_function(grid).reshape(xx.shape)

# Decision boundary (level 0), and margins at +/-1
plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors='purple')      # decision boundary
plt.contour(xx, yy, Z, levels=[-1, 1], linestyles='--', colors='gray') # margins

# Highlight support vectors
sv = clf.support_vectors_
plt.scatter(sv[:,0], sv[:,1], s=150, facecolors='none', edgecolors='red', linewidths=1.8, label='support vectors')

plt.xlabel("Number of 'buy' words")
plt.ylabel("Number of hyperlinks")
plt.title("Linear SVM: Spam detection (tiny dataset)")
plt.legend(loc='upper left')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("spam_svm.png", dpi=150)
plt.show()
