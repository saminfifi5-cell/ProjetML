#!/usr/bin/env python
# coding: utf-8

# # Projet : Modélisation du Risque de Crédit (Credit Scoring)
# ## Préparation des données et création de l'Arbre de décision fait de zéro
# 

# In[7]:


from pathlib import Path

import numpy as np
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
import os 
import matplotlib.pylab 
import matplotlib.pyplot 


file_path1 = Path(__file__).parent.parent / 'Loan_default.xlsx'

# Lire le fichier Excel
df = pd.read_excel(file_path1)


if 'LoanID' in df.columns:
    df = df.drop('LoanID', axis=1)

# Convertir les colonnes catégorielles en numériques
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = pd.factorize(df[col])[0]

print(df.head(20))

def gini(y):
    n = len(y) 
    if n == 0:
        return 0
    compte = Counter(y)
    somme = sum((count/n)**2 for count in compte.values())
    return 1 - somme

def entropy(y):
    n = len(y)
    if n == 0:
        return 0
    compte = Counter(y)
    somme = sum(-(count/n)*np.log2(count/n) for count in compte.values() if count > 0)
    return somme

def gain(y, y1, y2, criterion="gini"):
    if len(y1) == 0 or len(y2) == 0:
        return 0
    if criterion == "gini":
        H_y = gini(y)
        H_y1 = gini(y1)
        H_y2 = gini(y2)
    elif criterion == "entropy":
        H_y = entropy(y)
        H_y1 = entropy(y1)
        H_y2 = entropy(y2)
    else:
        raise ValueError("Unknown criterion")
    return H_y - ((len(y1)/len(y))*H_y1 + (len(y2)/len(y))*H_y2)

def best_split(X, y, criterion="gini"):
    n_samples, n_features = X.shape
    best_gain = -1
    best_feature = None
    best_threshold = None
    for feature in range(n_features):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold
            y_left = y[left_mask]
            y_right = y[right_mask]
            g = gain(y, y_left, y_right, criterion)
            if g > best_gain:
                best_gain = g
                best_feature = feature
                best_threshold = threshold
    return best_feature, best_threshold, best_gain

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, criterion="gini", max_depth=None, min_samples_split=2):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def most_common_label(self, y):
        compte = Counter(y)
        most_common = compte.most_common(1)[0][0]
        return most_common

    def build_tree(self, X, y, depth):
        n_samples = len(y)
        n_classes = len(np.unique(y))

        if self.max_depth is not None and depth >= self.max_depth:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        if n_samples < self.min_samples_split:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        if n_classes == 1:
            return Node(value=y[0])

        best_feature, best_threshold, best_gain = best_split(X, y, self.criterion)

        if best_gain == 0:
            leaf_value = self.most_common_label(y)
            return Node(value=leaf_value)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = X[:, best_feature] > best_threshold  

        left_child = self.build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth + 1)

        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def fit(self, X, y):
        self.root = self.build_tree(X, y, depth=0)
        return self

    def predict_sample(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.predict_sample(x, node.left)
        else:
            return self.predict_sample(x, node.right)

    def predict(self, X):
        return np.array([self.predict_sample(x, self.root) for x in X])


# ## Entraînement sur des données équilibrées (Undersampling)
# 
# **Le problème :** Notre dataset est fortement déséquilibré (88% de prêts remboursés, 12% de défauts). 
# 
# **La solution :** Pour forcer notre arbre "maison" à repérer les profils à risque, nous créons un sous-échantillon équilibré (50% de bons payeurs, 50% de fraudeurs). On l'évalue ensuite en privilégiant la métrique du **Rappel (Recall)** plutôt que l'Accuracy.

# In[9]:


import matplotlib.pylab 

# Équilibrage des données
df_bons = df[df['Default'] == 0]
df_mauvais = df[df['Default'] == 1]

df_bons_reduit = df_bons.sample(n=2000, random_state=42)
df_mauvais_reduit = df_mauvais.sample(n=2000, random_state=42)

# On mélange
df_equilibre = pd.concat([df_bons_reduit, df_mauvais_reduit]).sample(frac=1, random_state=42)

X = df_equilibre.drop('Default', axis=1).to_numpy()
y = df_equilibre['Default'].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Début de l'entraînement sur {len(X_train)} clients")

tree = DecisionTree(criterion="entropy", max_depth=2, min_samples_split=3)
tree.fit(X_train, y_train)

print("Entraînement terminé !\n")

predictions = tree.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Précision globale (Accuracy) : {accuracy:.2%}")

vrais_negatifs = np.sum((predictions == 0) & (y_test == 0))
faux_positifs = np.sum((predictions == 1) & (y_test == 0))
vrais_positifs = np.sum((predictions == 1) & (y_test == 1))
faux_negatifs = np.sum((predictions == 0) & (y_test == 1))

print("\n MATRICE DE CONFUSION ")
print(f"Bons payeurs correctement identifiés (Vrais Négatifs) : {vrais_negatifs}")
print(f"Bons payeurs refusés à tort (Faux Positifs) : {faux_positifs}")
print(f"Emprunteurs à risque correctement bloqués (Vrais Positifs) : {vrais_positifs}")
print(f"Emprunteurs à risque acceptés à tort (Faux Négatifs) : {faux_negatifs}")

if (vrais_positifs + faux_negatifs) > 0:
    recall = vrais_positifs / (vrais_positifs + faux_negatifs)
    print(f"\nRecall sur les défauts : {recall:.2%} des profils risqués ont été identifiés.")

print("\nRAPPEL DES LABELS :")
print("0 = Prêt remboursé (Pas de défaut)")
print("1 = Défaut de paiement")


# Sauvergarde de l'entrainement


# nom_fichier = '../models/arbre_fait_maison.pkl'

model = {
    'model': tree,      
    'mean': matplotlib.pylab.mean,
    'std': matplotlib.pylab.std
}

file_path2 = Path(__file__).parent.parent / 'models/arbre_fait_maison.pkl'

with open(file_path2, 'wb') as fichier:
    pickle.dump(tree, fichier)

print(f" arbre de décision sauvegardé sous le nom : {file_path2}\n")


# ## Le modèle de base (Scikit-Learn)
# 
# On comparons maintenant nos résultats avec la bibliothèque **Scikit-Learn**. 

# In[ ]:


from sklearn.tree import DecisionTreeClassifier


X_complet = df.drop('Default', axis=1).to_numpy()
y_complet = df['Default'].to_numpy()

X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(X_complet, y_complet, test_size=0.2, random_state=42)

print(f"Entraînement sur {len(X_train_sk)} clients...")

arbre_sk = DecisionTreeClassifier(
    criterion="entropy", 
    max_depth=6,              
    class_weight='balanced',  
    random_state=42
)

arbre_sk.fit(X_train_sk, y_train_sk)

predictions_sk = arbre_sk.predict(X_test_sk)

accuracy_sk = np.mean(predictions_sk == y_test_sk)
print(f"Précision globale (Accuracy) : {accuracy_sk:.2%}")

vrais_negatifs_sk = np.sum((predictions_sk == 0) & (y_test_sk == 0))
faux_positifs_sk = np.sum((predictions_sk == 1) & (y_test_sk == 0))
vrais_positifs_sk = np.sum((predictions_sk == 1) & (y_test_sk == 1))
faux_negatifs_sk = np.sum((predictions_sk == 0) & (y_test_sk == 1))

print("\nMATRICE DE CONFUSION SCIKIT-LEARN")
print(f"Bons payeurs acceptés (Vrais Négatifs) : {vrais_negatifs_sk}")
print(f"Bons payeurs refusés (Faux Positifs) : {faux_positifs_sk}")
print(f"Profils à risque bloqués (Vrais Positifs) : {vrais_positifs_sk}")
print(f"Profils à risque ratés (Faux Négatifs) : {faux_negatifs_sk}")

if (vrais_positifs_sk + faux_negatifs_sk) > 0:
    recall_sk = vrais_positifs_sk / (vrais_positifs_sk + faux_negatifs_sk)
    print(f"\nRappel (Recall) Scikit-Learn : {recall_sk:.2%} des profils risqués ont été identifiés.")


# ## Optimisation des Hyperparamètres (Grid Search)
# 
# On utilise un **Grid Search** avec Validation Croisée pour tester automatiquement toutes les combinaisons possibles et trouver le réglage qui maximise notre détection des défauts de paiement.

# In[ ]:


from sklearn.model_selection import GridSearchCV

print("Lancement du Grid Search")

arbre_de_base = DecisionTreeClassifier(class_weight='balanced', random_state=42)

parametres = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [4, 6, 8, 10, 12],
    'min_samples_split': [2, 10, 50, 100] 
}

recherche = GridSearchCV(
    estimator=arbre_de_base, 
    param_grid=parametres, 
    cv=3,      
    scoring='recall', 
    n_jobs=-1  
)

recherche.fit(X_train_sk, y_train_sk)

print("\n RÉSULTATS DU GRID SEARCH ")
print("Le meilleur réglage absolu est :")
print(recherche.best_params_)
print(f"Meilleur score de Rappel (Recall) estimé : {recherche.best_score_:.2%}")

