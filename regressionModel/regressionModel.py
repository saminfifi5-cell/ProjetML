import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt

#test push github

#df = pd.read_excel(r"C:\Users\swask\Desktop\Travail\coursinge\2iA\Introduction_Machine_Learning\Projet\Loan_default.xlsx")
#df = df.drop('LoanID', axis=1)
#for col in df.columns:
#    if df[col].dtype == 'object':
#        df[col] = pd.factorize(df[col])[0]

#print(df.head(20))


df = pd.read_csv('Loan_default.csv', encoding='latin1', sep=None, engine='python')
df = df.drop(columns=['LoanID'])
df = df.sample(n=50000, random_state=42).reset_index(drop=True)

#transformation des variables catégorielles

binary_cols = ['HasMortgage', 'HasDependents', 'HasCoSigner']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

multi_cols = ['Education', 'EmploymentType', 'MaritalStatus', 'LoanPurpose']
df = pd.get_dummies(df, columns=multi_cols, drop_first=False)

bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)

# Imputer les valeurs manquantes de LoanAmount par la médiane

df_majority = df[df['Default'] == 0]
df_minority = df[df['Default'] == 1]

df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

df_balanced = pd.concat([df_majority_downsampled, df_minority]).sample(frac=1, random_state=42)

print(df_balanced['Default'].value_counts())




X = df_balanced.drop(columns=['Default']).values 
y = df_balanced['Default'].values  

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

### normalisation ###
mean = X_train.mean(axis=0)
std  = X_train.std(axis=0)
std[std == 0] = 1

X_train = (X_train - mean) / std
X_test  = (X_test  - mean) / std

X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
X_test  = np.hstack([np.ones((X_test.shape[0], 1)), X_test])



### fonctions ###

# définition de la fonction sigmoïde pour la régression logistique #
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# définition de la fonction log-loss (perte) pour la régression logistique #

def compute_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-9, 1 - 1e-9)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


# implémentation de la descente de gradient pour la régression logistique avec régularisation L2 (Ridge) #

def gradient_descent(X, y, lr=0.01, n_iter=1000, accuracy_threshold=0.80, lambda_=0.01):
    

    n, p = X.shape
    w = np.zeros(p)
    history = []


    for i in range(n_iter):
        y_pred   = sigmoid(X @ w)
        reg = lambda_ * w
        reg[0] = 0
        gradient = (1 / n) * X.T @ (y_pred - y) + reg
        w       -= lr * gradient
        history.append(compute_loss(y, y_pred))
        #print(i)



    return w, history

# fonction de prédiction, on définit le seuil #

def predict(X, w, threshold=0.5):
    return (sigmoid(X @ w) >= threshold).astype(int)

# fonction d'évaluation #

def evaluate(y_true, y_pred, label=""):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc  = np.mean(y_true == y_pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"[{label}] Accuracy={acc:.3f} | Precision={prec:.3f} | Recall={rec:.3f} | F1={f1:.3f}")


### train et eval ###

w, loss_history = gradient_descent(X_train, y_train, lr=0.01, n_iter=1000)

evaluate(y_train, predict(X_train, w), label="Train")
evaluate(y_test,  predict(X_test,  w), label="Test ")

# plot de la loss #

plt.plot(loss_history)
plt.xlabel("Itérations")
plt.ylabel("Loss")
plt.title("Convergence de la descente de gradient")
plt.grid(True)
plt.show()