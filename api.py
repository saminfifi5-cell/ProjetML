# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

app = FastAPI()

# Chargement du modèle au démarrage
with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

w    = model['weights']
mean = model['mean']
std  = model['std']

# Schéma des données entrantes
class DossierClient(BaseModel):
    Age: int
    Income: float
    LoanAmount: float
    CreditScore: int
    MonthsEmployed: int
    NumCreditLines: int
    InterestRate: float
    LoanTerm: int
    DTIRatio: float
    HasMortgage: int       # 0 ou 1
    HasDependents: int     # 0 ou 1
    HasCoSigner: int       # 0 ou 1
    # ... ajouter toutes les features one-hot encodées

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

@app.post("/predict")
def predict(dossier: DossierClient):
    # Construire le vecteur de features
    x = np.array([list(dossier.dict().values())], dtype=float)

    # Normalisation avec les params du train
    x = (x - mean) / std

    # Ajout du biais
    x = np.hstack([np.ones((1, 1)), x])

    # Prédiction
    proba = sigmoid(x @ w)[0]
    decision = "Accordé" if proba >= 0.5 else "Refusé"

    return {
        "probabilite": round(float(proba), 3),
        "decision": decision
    }

# Lancer avec : uvicorn api:app --reload