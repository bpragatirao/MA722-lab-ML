# Machine Learning Lab Assignments 🤖

This repository hosts a variety of Machine Learning models, data preprocessing scripts, and evaluation metrics developed during lab sessions.

## 📌 Overview
From classical statistical learning to modern deep learning, these assignments document the end-to-end ML workflow: Data Cleaning -> Feature Engineering -> Model Training -> Evaluation.

## 🛠️ Tech Stack
* **Libraries:** Scikit-Learn, Pandas, NumPy, PyTorch/TensorFlow
* **Visualization:** Matplotlib, Seaborn
* **Tools:** Jupyter Notebooks, Google Colab

## 📂 Lab Assignments
* **Week 1: Linear & Logistic Regression** - Predicting continuous and categorical values.
* **Week 2: Decision Trees & Random Forests** - Exploring ensemble methods.
* **Week 3: Support Vector Machines (SVM)** - Kernel tricks and boundary optimization.
* **Week 4: Clustering** - Unsupervised learning with K-Means and PCA.

## 📊 Performance Summary
| Model | Accuracy | F1-Score | Best Parameters |
| :--- | :--- | :--- | :--- |
| Random Forest | 0.94 | 0.93 | n_estimators=100 |
| SVM (RBF) | 0.91 | 0.90 | C=1.0, gamma=auto |

## 🚀 Quick Usage (Python)
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
clf = RandomForestClassifier().fit(X_train, y_train)
print(f"Accuracy: {clf.score(X_test, y_test)}")
```
