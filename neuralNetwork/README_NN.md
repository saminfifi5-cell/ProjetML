# Documentation du Projet ML : Prédiction de Défaut de Paiement

## Présentation du Projet
Ce projet implémente un réseau de neurones (Perceptron Multicouche) pour prédire le risque de défaut de paiement sur des demandes de crédit. 

Afin de garantir la stabilité de l'environnement matériel (prévention des crashs de mémoire vive sous WSL/Linux) et d'assurer une reproductibilité stricte, le pipeline d'entraînement est segmenté en 5 notebooks distincts. L'exécution de bout en bout est gérée par un script d'orchestration conditionnel.

## Architecture du Pipeline
1. **1_Cleaner.ipynb** : Traitement des valeurs manquantes et traitement des valeurs aberrantes par winsorisation.
2. **2_Transformer.ipynb** : Encodage One-Hot des variables catégorielles et standardisation des variables continues.
3. **3_Pipeline.ipynb** : Équilibrage strict 50/50 des classes par sous-échantillonnage (Undersampling) et création des flux asynchrones `tf.data.Dataset`.
4. **4_Modelisation.ipynb** : Définition, compilation et entraînement du réseau de neurones Keras avec sauvegarde de l'artefact.
5. **5_Evaluation.ipynb** : Inférence sur les données de test, calcul du seuil d'acceptation optimal et exportation des métriques de performance.

Tous les artefacts générés (données nettoyées, encodeurs, modèles, métriques) sont strictement centralisés dans le répertoire automatisé `RN_sousdoss/`.

---

## 1. Prérequis et Installation

### 1.1. Prérequis Système
* Un environnement basé sur Linux (natif ou via Windows Subsystem for Linux - WSL).
* Python 3.10 ou supérieur installé sur le système.

### 1.2. Création de l'Environnement Virtuel
Il est impératif d'utiliser un environnement virtuel isolé pour éviter les conflits de dépendances système. Ouvrez un terminal à la racine du projet et exécutez :

```bash
python3 -m venv env
```

### 1.3. Activation de l'Environnement
Activez l'environnement virtuel fraîchement créé :

```bash
source env/bin/activate
```
*(Votre terminal doit désormais afficher `(env)` en préfixe de la ligne de commande).*

### 1.4. Installation des Dépendances
Installez les paquets requis pour l'analyse de données, l'apprentissage profond et l'orchestration des notebooks. Assurez-vous que l'environnement est actif :

```bash
pip install pandas numpy scikit-learn tensorflow keras jupyter nbconvert ipykernel
```

---

## 2. Exécution du Projet

L'exécution du projet ne se fait pas en ouvrant et en exécutant les notebooks manuellement. Elle est déléguée au script `Orchestrator.py`. 

Ce script lance chaque étape dans un sous-processus isolé. Dès qu'une étape est terminée, la mémoire vive (RAM) est intégralement purgée avant de passer à l'étape suivante.

### Lancement de l'Orchestrateur
Dans votre terminal, avec l'environnement virtuel activé, exécutez la commande suivante :

```bash
python Orchestrator.py
```

### Logique Conditionnelle
L'orchestrateur est conçu pour optimiser le temps de calcul. S'il détecte que le résultat d'une étape (artefact ou modèle) est déjà présent dans le répertoire `RN_sousdoss/`, il ignorera l'exécution du notebook correspondant. 

* **Pour forcer une réexécution complète :** Supprimez simplement le dossier `RN_sousdoss/` avant de lancer l'orchestrateur.
* **Pour forcer une réexécution partielle :** Supprimez le fichier spécifique correspondant à l'étape souhaitée (par exemple, supprimer `baseline_mlp.keras` forcera le ré-entraînement du modèle sans recalculer le nettoyage des données).

---

## 3. Analyse des Résultats

Une fois l'orchestration terminée avec succès, l'ensemble des résultats est disponible dans le répertoire `RN_sousdoss/`.

L'évaluation finale du modèle se trouve dans le fichier :
`RN_sousdoss/performance_finale.json`

**Indicateur métier principal :**
Dans le cadre de l'évaluation du risque de crédit, l'exactitude globale (Accuracy) n'est pas la métrique décisive en raison de la nature asymétrique du risque financier. L'analyse des performances doit se concentrer en priorité sur le **Recall de la Classe 1 (Défaut)**, qui indique la proportion réelle de mauvais payeurs que le modèle parvient à identifier et à écarter. Ici le modèle n'est pas très performant en raison de l'asymétrie des données. Il y a bien plus de personnes dont le prêt est accepté que de personnes faisant défaut. En essayant une distribution 50/50 les résultats restent sensiblement identiques (88% de précision pour 42% de recall) donc il a été choisi de garder ce modèle en réseau de neurone tel qu'il est actuellement.
```