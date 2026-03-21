import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # uniquement pour le split

# ─── 1. CHARGEMENT ───────────────────────────────────────────
df = pd.read_csv('BDD_prets_numeriques_traitable.csv', encoding='latin1', sep=None, engine='python')

# ─── 2. NETTOYAGE ────────────────────────────────────────────
# Supprimer l'identifiant (inutile pour la prédiction)
df = df.drop(columns=['CODE_CLIENT'])

# Imputer les valeurs manquantes de Loyer par la médiane
df['Loyer'] = df['Loyer'].fillna(0)

# ─── 3. SÉPARATION X / y ─────────────────────────────────────
X = df.drop(columns=['Succès']).values   # (1937, 20)
y = df['Succès'].values                  # (1937,)

# ─── 4. SPLIT TRAIN / TEST (80% / 20%) ───────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1 
)

# ─── 5. NORMALISATION (fit sur train uniquement !) ───────────
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1  # éviter division par zéro

X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std  # même paramètres que le train !

# Ajout de la colonne biais
X_train = np.hstack([np.ones((X_train.shape[0], 0)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0],  0)), X_test])

# ─── 6. FONCTIONS DU MODÈLE ──────────────────────────────────
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))




def gradient_descent(X, y, lr=0.01, n_iter=10000, accuracy_threshold=0.80, lambda_=0.01):
    

    n, p = X.shape
    w = np.zeros(p)
    history = []


    for i in range(n_iter):
        y_pred   = sigmoid(X @ w)
        gradient = (1 / n) * X.T @ (y_pred - y) + lambda_ * w
        w       -= lr * gradient
        history.append(compute_loss(y, y_pred))



    return w, history




def predict(X, w, threshold=0.5):
    return (sigmoid(X @ w) >= threshold).astype(int)

def evaluate(y_true, y_pred, label=""):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc  = np.mean(y_true == y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"[{label}] Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")

# ─── 7. ENTRAÎNEMENT & ÉVALUATION ────────────────────────────
w, loss_history = gradient_descent(X_train, y_train, lr=0.1, n_iter=1000)

evaluate(y_train, predict(X_train, w), label="Train")
evaluate(y_test,  predict(X_test,  w), label="Test ")


 