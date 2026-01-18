# Valve Condition Monitoring — Guide d’exécution (Mode 2 Streamlit)

Ce guide explique ce qui a été mis en place et comment lancer les tests et l’application web (Mode 2 : Streamlit) en local et via Docker.

---

## 1) Ce qui a été implémenté

- Pipeline ML dans `src/pipeline.py` :
  - Chargement du dataset final `FINAL_DATASET_READY_FOR_ML.csv`
  - Split séquentiel (Train = cycles 1→2000, Test = 2001→N)
  - Chargement des artefacts sauvegardés : `models/random_forest_model.pkl` (modèle), `models/scaler.pkl` (StandardScaler)
  - Prédiction pour un cycle donné (classe + probabilités si dispo)
  - Extraction Top 5 features importantes (RandomForest)
- Application Streamlit dans `app/app.py` (Mode 2) :
  - Entrée : numéro de cycle
  - Affichage : classe prédite, probabilités par classe, top 5 features et leurs valeurs pour ce cycle
- Tests unitaires dans `tests/test_pipeline.py` :
  - Dataset OK : alignement X/y, absence de NaN/Inf, 1945 features
  - Split OK : respect du découpage 1→2000 / 2001→N
  - Standardisation OK : transform sur train/test, dimensions inchangées, train ≈ centré
  - Prédiction OK : une ligne → une classe ∈ {100, 90, 80, 73} ; somme des probas ≈ 1
- Packaging : `requirements.txt` mis à jour (scikit-learn, joblib, streamlit, pytest)
- Conteneur Mode 2 : `Dockerfile` pour lancer `app/app.py` avec les artefacts et le dataset

---

## 2) Prérequis

- Windows + PowerShell
- Python 3.11+ recommandé
- Dataset présent : `FINAL_DATASET_READY_FOR_ML.csv`
- Artefacts présents : `models/random_forest_model.pkl`, `models/scaler.pkl`

---

## 3) Installation des dépendances (local)

Dans le dossier du projet :

```powershell
pip install -r requirements.txt
```

Si vous utilisez un venv spécifique, activez-le avant (exemples) :

```powershell
# Exemples
# & "C:\Path\to\myvenv\Scripts\Activate.ps1"
# ou
# & ".venv\Scripts\Activate.ps1"
```

---

## 4) Lancer les tests (pytest)

Exécuter les tests unitaires qui valident le pipeline :

```powershell
python -m pytest -q
```

Si vous devez cibler un Python précis (ex : venv) :

```powershell
& "C:\Users\steph\OneDrive\Bureau\cour M1 ynov\machine learning\projet final\condition+monitoring+of+hydraulic+systems\myvenv\Scripts\python.exe" -m pytest -q
```
### ✅ Résultat attendu : 6 tests passent

**Ce qui a été fixé :**

Le dataset original a 1947 colonnes : `[cycle | 1945 features | valve_condition]`

Avant le fix :
- En excluant uniquement `valve_condition`, on gardait `cycle` (l'identifiant)
- Résultat : X avait 1946 colonnes (cycle + 1945 features) ❌
- Le scaler (entraîné sur 1945 features) rejetait cette entrée

Après le fix :
- En excluant AUSSI la colonne `cycle` (qui est juste un identifiant, pas une feature)
- Résultat : X a exactement 1945 colonnes ✅
- Le scaler et le modèle fonctionnent correctement

**Les 6 tests valident :**

| Test | Vérifie |
|------|---------|
| `test_dataset_alignment` | X et y même longueur, 1945 features exactement, pas de NaN/Inf |
| `test_features_no_nan_inf` | Absence de valeurs NaN ou Inf dans les features |
| `test_split_sequential` | Split correct : train = cycles 1→2000, test = 2001→N |
| `test_standardization_transform` | Scaler transforme train/test, dimensions préservées |
| `test_prediction_single_cycle` | Prédiction d'un cycle → classe ∈ {100, 90, 80, 73}, probas ≈ 1 |
| `test_model_artifacts_exist` | Fichiers RF et scaler présents dans `models/` |

**Les warnings (non-critiques) :**

```
UserWarning: X does not have valid feature names, but StandardScaler was fitted with feature names
```

C'est un avertissement inoffensif : le scaler a été entraîné avec les noms de colonnes pandas, mais on lui passe parfois des numpy arrays sans noms. Le scaler fonctionne quand même correctement.
---

## 5) Lancer l’app Streamlit (Mode 2) — Local

Démarrer l’application :

```powershell
streamlit run app/app.py
```

Ouvrir ensuite : http://localhost:8501

Workflow :
- Saisir un numéro de cycle (ex : 1500)
- Cliquer « Predict »
- Voir la classe prédite, les probabilités, et les top 5 features (avec leurs valeurs pour ce cycle)

---

## 6) Docker — Build & Run (Mode 2)

### Option A : Docker seul

Construire l'image :

```powershell
docker build -t valve-monitor:app .
```

Lancer le conteneur :

```powershell
docker run --rm -p 8501:8501 valve-monitor:app
```

Ouvrir : http://localhost:8501

L'image embarque :
- `src/`, `app/`, `models/`
- `FINAL_DATASET_READY_FOR_ML.csv`
- Dépendances de `requirements.txt`

### Option B : Docker Compose (Recommandé)

**Configuration :**
Le fichier `docker-compose.yml` est configuré avec :
- 🐋 Build automatique depuis le Dockerfile
- 🌐 Port 8501 exposé pour accéder à l'app Streamlit
- 📁 Volumes montés pour le développement (modification du code sans rebuild)
- 🔄 Restart automatique en cas d'erreur
- 💚 Healthcheck pour vérifier la disponibilité de l'application

**Démarrer l'application :**

```powershell
docker-compose up -d
```

L'option `-d` lance le conteneur en arrière-plan (mode détaché).

**Accéder à l'application :**

Ouvrir : http://localhost:8501

**Voir les logs :**

```powershell
docker-compose logs -f
```

Le flag `-f` permet de suivre les logs en temps réel (Ctrl+C pour quitter).

**Arrêter l'application :**

```powershell
docker-compose down
```

Cette commande arrête et supprime le conteneur (les volumes/images restent).

**Redémarrer après modification :**

```powershell
docker-compose restart
```

**Rebuild après modification du Dockerfile ou requirements.txt :**

```powershell
docker-compose up -d --build
```

**Avantages de Docker Compose :**
- Configuration simplifiée dans un seul fichier YAML
- Gestion facile du cycle de vie (up/down/restart)
- Volumes montés pour développement itératif sans rebuild constant
- Healthcheck intégré pour monitoring
- Commandes courtes et mémorables

---

## 7) Structure du projet

- `src/` : pipeline et utilitaires
- `app/` : application Streamlit (Mode 2)
- `models/` : artefacts (RF + scaler)
- `tests/` : tests unitaires
- `requirements.txt` : dépendances
- `Dockerfile` : conteneur Mode 2
- `docker-compose.yml` : orchestration Docker simplifiée
- `FINAL_DATASET_READY_FOR_ML.csv` : dataset final prêt ML

---

## 8) Dépannage (tips)

- Problème de commande `pytest` non trouvée : utilisez `python -m pytest -q`
- `streamlit` non trouvé : `pip install -r requirements.txt` (vérifiez le venv actif)
- Dataset manquant : placez `FINAL_DATASET_READY_FOR_ML.csv` à la racine du projet
- Artefacts manquants : assurez-vous que `models/random_forest_model.pkl` et `models/scaler.pkl` existent
- Cycle non trouvé : choisissez un cycle valide 1..N (N = nombre de lignes du dataset)

---

## 9) Test rapide en ligne de commande (optionnel)

Le pipeline peut prédire depuis un script (exemple) :

```powershell
python src/pipeline.py
```

Cela affichera une prédiction et des infos pour le cycle 1500 (config par défaut dans `__main__`).

---

## 10) Prochaines étapes (si besoin)

- Mode 1 (train + evaluate) : ajouter `src/train.py` et `src/evaluate.py`, étendre Docker pour lancer ces scripts
- Ajouter d'autres services dans `docker-compose.yml` (base de données, monitoring, etc.)
- Monitoring en production (logs, métriques, alertes)
