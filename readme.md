Capteurs de PRESSION (100 Hz) : PS
Capteur de PUISSANCE MOTEUR (100 Hz): EPS
Capteurs de DÉBIT (10 Hz) : FS
Capteurs de TEMPÉRATURE (1 Hz) : TS
Capteur de VIBRATION (1 Hz) : VS
🟡 Capteurs VIRTUELS (calculés) :CE.txt → Cooling Efficiency , CP.txt → Cooling Power , SE.txt → Efficiency factor

a utiliser : ts_features_global_aggregation, PS_all_cycles_windowed_mean_min_max, 


# 📚 RÉSUMÉ COMPLET : PIPELINE DE PRÉPARATION DES DONNÉES

## 🎯 Objectif Global

Préparer un **dataset complet** pour prédire la **condition de la valve hydraulique** à partir de **13 capteurs multi-fréquences** mesurant pression, débit, température et vibration.

---

## 📊 1. Données Brutes Initiales

### Capteurs Disponibles (13 au total)

| Type | Capteurs | Fréquence | Points/cycle | Fichiers |
|------|----------|-----------|--------------|----------|
| **Pression** | PS1, PS2, PS3, PS5, PS6 (PS4 exclu) | 100 Hz | 6000 | PS1.txt, PS2.txt, etc. |
| **Pression externe** | EPS1 | 100 Hz | 6000 | EPS1.txt |
| **Débit** | FS1, FS2 | 10 Hz | 600 | FS1.txt, FS2.txt |
| **Température** | TS1, TS2, TS3, TS4 | 1 Hz | 60 | TS1.txt, TS2.txt, etc. |
| **Vibration** | VS1 | 1 Hz | 60 | VS1.txt |

### Labels
- **Fichier** : profile.txt (colonne 1 = valve_condition)
- **Classes** : 100 (optimal), 90, 80, 73 (défaillant)
- **Cycles** : 2205 cycles complets de 60 secondes chacun

---

## 🔧 2. Stratégies de Réduction par Fréquence

### ⚡ Capteurs Haute Fréquence (100 Hz) : **FENÊTRAGE**

**Capteurs concernés : PS (5 capteurs) + EPS (1 capteur)**

- **Problème** : 6000 points/cycle = trop de données brutes
- **Solution** : Fenêtrage de **100 points** (= 1 seconde)
- **Résultat** : 60 fenêtres/cycle
- **Statistiques par fenêtre** : mean, min, max, std (4 stats)
- **Formule** : 60 fenêtres × 4 stats = **240 features/capteur**

**Réduction :**
- PS : 6000 points → 1200 features (5 capteurs × 240)
- EPS : 6000 points → 240 features (1 capteur × 240)

**Fichiers créés :**
- `PS_all_cycles_windowed_mean_min_max.txt` (2205 × 1200)
- `EPS_features_windowed_100pts.txt` (2205 × 240)

---

### 🌊 Capteurs Fréquence Moyenne (10 Hz) : **FENÊTRAGE**

**Capteurs concernés : FS (2 capteurs)**

- **Problème** : 600 points/cycle = moyennement volumineux
- **Solution** : Fenêtrage de **10 points** (= 1 seconde)
- **Résultat** : 60 fenêtres/cycle
- **Statistiques par fenêtre** : mean, min, max, std (4 stats)
- **Formule** : 60 fenêtres × 4 stats = **240 features/capteur**

**Réduction :**
- FS : 600 points → 480 features (2 capteurs × 240)

**Fichier créé :**
- `FS_features_windowed_10pts.txt` (2205 × 480)

---

### 🐌 Capteurs Basse Fréquence (1 Hz) : **AGRÉGATION GLOBALE**

**Capteurs concernés : TS (4 capteurs) + VS (1 capteur)**

- **Problème** : 60 points/cycle = déjà compact
- **Constat** : Fenêtrer 1 point par fenêtre n'a aucun sens !
- **Solution** : **Agrégation globale par cycle**
- **Statistiques globales** : mean, min, max, std, **slope** (5 stats)
- **Formule** : 5 stats globales = **5 features/capteur**

**Réduction :**
- TS : 60 points → 20 features (4 capteurs × 5)
- VS : 60 points → 5 features (1 capteur × 5)

**Fichiers créés :**
- `TS_features_global_aggregation.txt` (2205 × 20)
- `VS_features_global_aggregation.txt` (2205 × 5)

**Avantage de l'agrégation** :
- ✅ Beaucoup plus simple (5 vs 240 features)
- ✅ Capture l'essence : tendance + variabilité + amplitude
- ✅ Pas de surapprentissage avec trop de features
- ✅ Pente (slope) ajoute l'information de tendance temporelle

---

## 🔗 3. Concaténation : Dataset Master

### Principe Fondamental
**Tous les capteurs ont enregistré les MÊMES cycles en MÊME TEMPS !**

```
Cycle Physique 1 (60 secondes) :
├─ PS ligne 0  : Mesures de pression
├─ FS ligne 0  : Mesures de débit
├─ EPS ligne 0 : Mesure de pression externe
├─ TS ligne 0  : Mesures de température
└─ VS ligne 0  : Mesure de vibration
```

### Problème Rencontré
- PS avait **2206 cycles** au lieu de **2205**
- **Solution** : Tronquer PS aux 2205 premiers cycles

### Concaténation Horizontale
```python
master_dataset = [PS | FS | EPS | TS | VS]
                 ↓    ↓    ↓    ↓    ↓
               1200 + 480 + 240 + 20 + 5 = 1945 features
```

**Fichier créé :**
- `MASTER_dataset_all_sensors.txt` (2205 × 1945)

---

## 🏷️ 4. Extraction des Labels

### Source
- **Fichier** : profile.txt (contient plusieurs colonnes)
- **Colonne utilisée** : feature_1 = valve_condition

### Extraction
```python
valve_condition = profile_df['feature_1']
```

**Classes :**
- **100** : Valve en condition optimale
- **90** : Légère dégradation
- **80** : Dégradation moyenne
- **73** : Valve défaillante

**Fichier créé :**
- `valve_condition.txt` (2205 × 1)

---

## 🎉 5. Dataset Final Complet

### Concaténation Finale
```python
final_dataset = [master_dataset | valve_condition]
                     1945          +       1       = 1946 colonnes
```

### Structure du Dataset Final

| Colonnes | Contenu | Description |
|----------|---------|-------------|
| 0 - 1199 | PS features | 5 capteurs de pression (fenêtrés) |
| 1200 - 1679 | FS features | 2 capteurs de débit (fenêtrés) |
| 1680 - 1919 | EPS features | 1 capteur pression externe (fenêtré) |
| 1920 - 1939 | TS features | 4 capteurs température (agrégés) |
| 1940 - 1944 | VS features | 1 capteur vibration (agrégé) |
| **1945** | **Label** | **valve_condition (100, 90, 80, 73)** |

### Caractéristiques Finales
- **Shape** : 2205 cycles × 1946 colonnes
- **Features** : 1945 (tous les capteurs combinés)
- **Label** : 1 (condition de la valve)
- **Format** : Prêt pour l'apprentissage supervisé

**Fichier créé :**
- `FINAL_dataset_with_labels.txt` (2205 × 1946)

---

## 📈 6. Résumé des Réductions

| Capteur(s) | Original | Final | Réduction |
|------------|----------|-------|-----------|
| PS (5) | 30,000 pts | 1200 features | **25x** |
| EPS (1) | 6,000 pts | 240 features | **25x** |
| FS (2) | 1,200 pts | 480 features | **2.5x** |
| TS (4) | 240 pts | 20 features | **12x** |
| VS (1) | 60 pts | 5 features | **12x** |
| **TOTAL** | **37,500 pts** | **1945 features** | **~19x** |

---

## 🚀 7. Prochaines Étapes ML

### Pipeline d'Entraînement

1. **Séparer X et y**
   ```python
   X = final_dataset.iloc[:, :-1]  # Features (1945 colonnes)
   y = final_dataset.iloc[:, -1]   # Labels (valve_condition)
   ```

2. **Normalisation**
   ```python
   from sklearn.preprocessing import StandardScaler
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)
   ```

3. **Split Train/Test**
   ```python
   from sklearn.model_selection import train_test_split
   X_train, X_test, y_train, y_test = train_test_split(
       X_scaled, y, test_size=0.2, random_state=42, stratify=y
   )
   ```

4. **Entraînement du Modèle**
   ```python
   from sklearn.ensemble import RandomForestClassifier
   model = RandomForestClassifier(n_estimators=100, random_state=42)
   model.fit(X_train, y_train)
   ```

5. **Évaluation**
   ```python
   from sklearn.metrics import classification_report, confusion_matrix
   y_pred = model.predict(X_test)
   print(classification_report(y_test, y_pred))
   ```

---

## ✅ Ce que tu as accompli

✨ **Pipeline complet de feature engineering multi-fréquences**
✨ **13 capteurs** → **1945 features cohérentes**
✨ **Fenêtrage intelligent** pour capteurs haute fréquence
✨ **Agrégation globale** pour capteurs basse fréquence
✨ **Dataset final unifié** avec features + labels alignés
✨ **Réduction massive** (~19x) tout en préservant l'information essentielle
✨ **Prêt pour l'entraînement** de modèles ML de classification

🎯 **Tu as transformé 37,500 points bruts en 1,945 features intelligentes !**


Étape 0 — Verrouiller ton dataset final

Objectif : être sûr que ton X (features) et ton y (valve condition) sont clean.

À vérifier avant tout :

1 ligne = 1 cycle (index ou colonne cycle)

y = colonne 2 de profile (100/90/80/73)

pas de valeurs manquantes

pas de colonnes constantes (ex : PS4 déjà retiré)

toutes les colonnes sont numériques

Livrable :

un tableau final “prêt ML” (features + label) + un mini résumé (nb cycles, nb features, classes de y)

Étape 1 — Visualisations utiles (pour comprendre et pour ton rapport)

Objectif : montrer que tu comprends ce que ton modèle va apprendre.

A. Distribution des classes (valve condition)

Compter combien de cycles sont 100, 90, 80, 73

Visualiser (bar chart)

Pourquoi : ça te dit si ton dataset est déséquilibré → impact sur les métriques.

B. Visualiser quelques features “importantes” par classe

Choisis 5–10 features compréhensibles :

ex. PS1_mean_w10, PS2_std_w9, FS1_mean_w12, TS4_slope, EPS1_std_w8

Regarde :

distribution par classe (boxplots / violins)

corrélation simple (heatmap sur un sous-ensemble)

Pourquoi : tu vérifies que certaines variables séparent bien les classes.

C. PCA / t-SNE (optionnel mais très parlant)

Projeter ton dataset en 2D et colorer par classe
Pourquoi : si les classes se séparent déjà, ton modèle va bien marcher.

Livrable :

2–3 figures propres + 3 phrases d’interprétation max (courtes et claires)

Étape 2 — Normalisation / Standardisation (préparer les données)

Objectif : rendre les features comparables et éviter qu’une feature domine juste par son échelle.

Quelle normalisation choisir ?

Si tu utilises un modèle basé sur distance (SVM, KNN, régression logistique) → standardisation indispensable

Si tu utilises Random Forest / XGBoost → pas obligatoire, mais ça reste propre si tu compares des modèles

Très important : éviter la fuite de données

Tu dois :

calculer la normalisation uniquement sur le train (2000 premiers cycles)

appliquer ensuite la même transformation au test (cycles restants)

Livrable :

une section “Prétraitement” dans ton rapport expliquant que la standardisation est fit sur le train uniquement.

Étape 3 — Split Train/Test comme tu l’as demandé

Objectif : respecter ta contrainte “2000 premiers cycles = train, reste = test final”.

Ton split :

Train : cycles 1 → 2000

Test final : cycles 2001 → fin

Attention importante (à mentionner)

Ce dataset suit souvent une évolution “dans le temps” (dégradation progressive).
Ton split par ordre des cycles simule une situation réaliste :

tu entraînes sur le passé

tu testes sur le futur

👉 C’est très défendable, mais ça peut être plus dur qu’un split aléatoire.

Livrable :

afficher les tailles train/test

vérifier que les classes dans le test existent bien (ex : si une classe n’apparaît pas du tout dans le test, c’est un souci)

Étape 4 — Choisir un modèle et construire une baseline

Objectif : avoir un modèle simple qui marche, interprétable, facile à défendre.

Modèle recommandé pour ton cas

Random Forest (excellent baseline)

robuste

interprétable (importance des features)

marche très bien sur features agrégées

Option si tu veux comparer

Logistic Regression (avec standardisation)

SVM (avec standardisation)

Gradient Boosting (si tu veux pousser un peu)

Livrable :

un modèle baseline entraîné

un tableau de résultats sur le test final

Étape 5 — Évaluer le modèle sur le test final

Objectif : répondre exactement à “Évaluez sur l’échantillon de test final”.

Métriques à produire

Accuracy (ok mais insuffisant seul)

Matrice de confusion (indispensable)

Precision / Recall / F1 par classe (surtout si déséquilibre)

(optionnel) Balanced accuracy

Ce que tu dois commenter

quelles classes sont le plus confondues (ex : 90 vs 80)

est-ce que le modèle détecte bien les cas “73” (quasi panne)

est-ce que “100” est très bien reconnu

Livrable :

matrice de confusion + classification report + 5 lignes d’analyse

Étape 6 — Interprétation : quelles features comptent ?

Objectif : faire la partie “ingénierie” + gagner des points.

Deux choses super efficaces :

Top 15 importances (Random Forest)

regrouper par capteur (ex : PS vs FS vs TS)

Ce que tu veux montrer :

les transitoires (fenêtres autour des commutations) sont discriminants

les capteurs de débit et puissance apportent du signal

TS/VS apportent du contexte

Livrable :

un bar chart des top features + explication courte

Étape 7 — Tests unitaires

Objectif : prouver que ton code est fiable et ré-exécutable.

Tu ne testes pas “le ML” directement, tu testes les composants critiques.

Tests utiles (et réalistes)

Test d’alignement des cycles

vérifier que X et y ont le même nombre de lignes

vérifier que le cycle i correspond bien au label i

Test de features

pas de NaN

pas d’inf

dimensions attendues (nb de colonnes)

Test de prétraitement

standardizer fit sur train seulement

transformer(train) et transformer(test) fonctionnent

Test de prédiction

un cycle donné renvoie une prédiction parmi {100, 90, 80, 73}

Livrable :

une suite de tests qui passe en local et en Docker

Étape 8 — Containerisation (Docker)

Objectif : permettre à quelqu’un de lancer ton projet en 2 commandes.

À inclure dans l’image

ton code

ton modèle entraîné (ou un script qui l’entraîne)

les dépendances (requirements)

un point d’entrée clair

Deux modes possibles

Mode 1 : “train + evaluate”

Mode 2 : “run app streamlit”

Livrable :

Dockerfile + README “comment exécuter”

Étape 9 — Application Web Streamlit

Objectif : prédire l’état de la valve à partir d’un numéro de cycle.

Interface simple

input : numéro de cycle (ex : 1500)

bouton : “Predict”

output :

classe prédite (100/90/80/73)

(bonus) probas par classe

(bonus) 5 features principales de ce cycle

Point clé

Ton app doit :

charger le dataset final (features)

récupérer la ligne correspondant au cycle

appliquer la même normalisation (si utilisée)

appeler le modèle

afficher résultat

Livrable :

une app reproductible, claire, qui marche avec Docker

Étape 10 — Structure recommandée du projet (pour être propre)

Tu vas avoir quelque chose comme :

data/ (dataset final ou chemin)

src/ (prétraitement, modèle, prédiction)

tests/

models/ (modèle sauvegardé)

app/ (streamlit)

Dockerfile

README.md

Livrable :

structure claire + instructions d’exécution

Par quoi tu commences maintenant ?

Dans l’ordre le plus efficace :

Visualisations essentielles (classes + quelques features)

Split 2000 / reste

Standardisation (fit train, apply test)

Random Forest baseline

Évaluation + confusion matrix

Sauvegarder modèle + pipeline

Streamlit

Tests unitaires

Docker + README