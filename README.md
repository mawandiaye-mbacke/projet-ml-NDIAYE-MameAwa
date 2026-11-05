# Projet 2 — Détecter des attaques réseau avec le Machine Learning

**Nom / Prénom : NDIAYE Mame Awa** — Master RESI

Un système de détection d'intrusion (IDS) doit décider, pour chaque flux réseau, s'il est **normal (0)** ou une **attaque (1)** : scan de ports, force brute SSH/Telnet, déni de service. Ce dépôt entraîne et compare trois modèles, puis livre le meilleur dans `models/model.joblib`.

## Lancer le projet

```bash
pip install -r requirements.txt
python src/train.py                       # exploration, comparaison des modèles, enregistrement du meilleur
python src/predict.py data/flux_train.csv # exemple de prédiction
```

Utilisation de la fonction de prédiction :

```python
import pandas as pd
from src.predict import predict
predictions = predict(df)   # df : mêmes colonnes que flux_train.csv, sans 'attaque' -> liste de 0/1
```

## Démarche

1. **Exploration** : 3 000 flux, 9 variables ; 90 valeurs manquantes dans `duration` ; 72,2 % de flux normaux et **27,8 % d'attaques** (classes déséquilibrées : on juge les modèles sur le **F1 de la classe attaque**, pas sur l'accuracy).
2. **Prétraitement dans un `Pipeline`** : imputation par la médiane (`duration`), standardisation des variables numériques, encodage one-hot de `protocol` (`handle_unknown="ignore"` pour accepter un protocole inconnu).
3. **Trois modèles comparés** sur un jeu de validation stratifié (20 %, soit 600 flux) : régression logistique, arbre de décision (profondeur 6), forêt aléatoire (300 arbres).
4. **Choix du meilleur** sur le F1 de la classe attaque, vérifié par validation croisée à 5 plis, puis ré-entraînement sur les 3 000 flux et enregistrement dans `models/model.joblib`. Le fichier contient tout le `Pipeline` : prétraitement + modèle.

## Structure du dépôt

```
├── README.md
├── requirements.txt
├── data/flux_train.csv
├── src/train.py
├── src/predict.py
├── models/model.joblib
└── figures/matrice_confusion.png
```

---
## Résultats

**Nom / Prénom : NDIAYE Mame Awa**

Scores sur le jeu de validation (600 flux, dont 167 attaques) :

| Modèle | F1 (attaque) | Rappel (attaque) | Précision (attaque) | F1 en validation croisée (5 plis) |
|---|---|---|---|---|
| Régression logistique | 0,858 | 0,760 | 0,984 | 0,863 ± 0,033 |
| Arbre de décision | 0,951 | 0,934 | 0,969 | 0,941 ± 0,010 |
| **Forêt aléatoire** | **0,982** | **0,976** | **0,988** | **0,959 ± 0,009** |

**Modèle retenu et pourquoi :** la **forêt aléatoire**. Elle a le meilleur F1 de la classe attaque (0,982 en validation, 0,959 ± 0,009 en validation croisée), le meilleur rappel et la meilleure précision. C'est aussi le modèle le plus stable (écart-type le plus faible). La régression logistique rate beaucoup d'attaques (rappel 0,76) car la frontière entre normal et attaque n'est pas linéaire : une attaque se reconnaît à des **combinaisons** de signes (beaucoup de connexions vers le même hôte **et** un fort taux de SYN, ou des échecs de connexion **et** peu d'octets reçus), que les arbres captent naturellement.

**Analyse des erreurs (faux négatifs / faux positifs) :**

Matrice de confusion de la forêt sur le jeu de validation (voir `figures/matrice_confusion.png`) :

| | prédit normal | prédit attaque |
|---|---|---|
| **vrai normal** | 431 | 2 (faux positifs) |
| **vraie attaque** | **4 (faux négatifs)** | 163 |

- **4 attaques ratées sur 167** (2,4 %). Ces 4 flux ressemblent en tout point à du trafic normal : aucun échec d'authentification, 3 à 7 connexions vers le même hôte (contre 17 en médiane pour les attaques détectées), un serveur qui répond normalement (2 200 à 2 300 octets reçus pour 3 d'entre eux) et des ports courants (22, 80, 443, 8080). Ce sont des attaques **discrètes** (lentes, qui imitent un usage légitime), ou des étiquettes erronées.
- **Les faux négatifs sont le pire cas en sécurité** : une attaque non détectée passe sans alerte ni blocage, et l'attaquant peut continuer (vol de données, prise de contrôle, propagation) sans que personne ne le sache. Une fausse alerte, elle, coûte seulement quelques minutes d'analyse.
- **2 fausses alertes sur 433 flux normaux** (0,5 %). L'une ressemble exactement à un scan (taux de SYN 0,90, 46 connexions vers le même hôte, port inhabituel 8730) et pourrait être une erreur d'étiquette. L'autre est une connexion SSH avec **9 échecs d'authentification** : un utilisateur qui se trompe de mot de passe, qu'on ne peut pas distinguer d'une force brute avec ces seules variables.
- **Piste d'amélioration** : pour rater encore moins d'attaques, on peut abaisser le seuil de décision (par exemple 0,3 au lieu de 0,5), ce qui augmente le rappel au prix de quelques fausses alertes de plus.
