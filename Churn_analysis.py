# ======================
# 1. Import des librairies
# ======================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score, precision_recall_curve
)


# ======================
# 2. Charger les données
# ======================
df = pd.read_excel("C:/Users/olivi/OneDrive/Bureau/Data science/Junior Data Analyst/Projets/Portfolio/Churn_prediction/churn-bigml.xlsx")

print("Shape:", df.shape)
print(df.head())


# ======================
# 3. Aperçu des données
# ======================
print(df.info())
print(df['Churn'].value_counts(normalize=True))

sns.countplot(data=df, x="Churn")
plt.title("Distribution de la variable cible (Churn)")
plt.show()


# ======================
# 4. Préparation des données
# ======================

# On enlève des colonnes identifiants inutiles
df = df.drop(columns=["Area code","Total day charge","Total eve charge", "Total night charge", "Total intl charge"], axis=1)


# Encodage de la variable cible (Churn)
df["Churn"] = df["Churn"].map({True: 1, False: 0})

# Encodage binaire des colonnes Yes/No
df["International plan"] = df["International plan"].map({"Yes": 1, "No": 0})
df["Voice mail plan"] = df["Voice mail plan"].map({"Yes": 1, "No": 0})

### X et y -- vale counts déjà fait ligne 33 ?? Normalize ?
X = df.drop(["Churn","State"], axis=1)
y = df["Churn"]

print("Dimensions X :", X.shape)
print("Dimensions y :", y.shape)
print("Répartition de la cible :\n", y.value_counts(normalize=True))



#### Séparation train / test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("\nSplit effectué :")
print("Train :", X_train.shape, " Test :", X_test.shape)

# Normalisation
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

### On garde une version DataFrame pour RandomForest
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.select_dtypes(include=np.number).columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.select_dtypes(include=np.number).columns)

print("\nLes données sont prêtes pour la modélisation")



# ======================
# 5. Modèle Logistic Regression
# ======================
logreg = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
logreg.fit(X_train_scaled, y_train)

y_pred_log = logreg.predict(X_test_scaled)
y_proba_log = logreg.predict_proba(X_test_scaled)[:, 1]

print("Classification Report - Logistic Regression")
print(classification_report(y_test, y_pred_log))


# ======================
# 6. Modèle Random Forest
# ======================
rf = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced")
rf.fit(X_train_scaled_df, y_train)

y_pred_rf = rf.predict(X_test_scaled_df)
y_proba_rf = rf.predict_proba(X_test_scaled_df)[:, 1]

print("Classification Report - Random Forest")
print(classification_report(y_test, y_pred_rf))


# ======================
# 7. ROC / AUC comparée
# ======================
roc_log = roc_auc_score(y_test, y_proba_log)
roc_rf = roc_auc_score(y_test, y_proba_rf)

fpr_log, tpr_log, _ = roc_curve(y_test, y_proba_log)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_proba_rf)

plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f"LogReg (AUC={roc_log:.2f})")
plt.plot(fpr_rf, tpr_rf, label=f"RandomForest (AUC={roc_rf:.2f})")
plt.plot([0,1],[0,1],"k--")
plt.xlabel("Taux de faux positifs")
plt.ylabel("Taux de vrais positifs")
plt.title("Courbe ROC")
plt.legend()
plt.grid()
plt.show()


# ======================
# 8. Optimisation du seuil (Logistic Regression)
# ======================
thresholds = np.arange(0.1, 0.91, 0.05)
precisions, recalls, f1s = [], [], []

for t in thresholds:
    y_pred_thresh = (y_proba_log >= t).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh))
    recalls.append(recall_score(y_test, y_pred_thresh))
    f1s.append(f1_score(y_test, y_pred_thresh))

plt.figure(figsize=(10,6))
plt.plot(thresholds, precisions, label="Precision")
plt.plot(thresholds, recalls, label="Recall")
plt.plot(thresholds, f1s, label="F1 Score")
plt.axvline(0.5, color="grey", linestyle="--", label="Seuil par défaut")
plt.xlabel("Seuil")
plt.ylabel("Score")
plt.title("Optimisation du seuil - Logistic Regression")
plt.legend()
plt.grid()
plt.show()

best_threshold = thresholds[np.argmax(f1s)]
print("Seuil optimal (F1 max):", best_threshold)


# 8bis. Optimisation du seuil - Random Forest
# ======================
# Courbe précision / rappel / F1 en fonction du seuil
thresholds_rf = np.arange(0.05, 0.951, 0.01)
precisions_rf, recalls_rf, f1s_rf = [], [], []

for t in thresholds_rf:
    y_pred_rf_t = (y_proba_rf >= t).astype(int)
    precisions_rf.append(precision_score(y_test, y_pred_rf_t, zero_division=0))
    recalls_rf.append(recall_score(y_test, y_pred_rf_t))
    f1s_rf.append(f1_score(y_test, y_pred_rf_t))

# Plot
plt.figure(figsize=(10,6))
plt.plot(thresholds_rf, precisions_rf, label="Précision")
plt.plot(thresholds_rf, recalls_rf, label="Rappel")
plt.plot(thresholds_rf, f1s_rf, label="F1 Score")
plt.axvline(0.5, color="grey", linestyle="--", label="Seuil par défaut (0.5)")
plt.xlabel("Seuil")
plt.ylabel("Score")
plt.title("Optimisation du seuil - Random Forest")
plt.legend()
plt.grid()
plt.show()

# Seuil RF optimal selon F1
best_threshold_rf = thresholds_rf[np.argmax(f1s_rf)]
print("Seuil optimal RF (F1 max):", best_threshold_rf)

# ======================
# 9. Matrice de confusion avec seuil optimisé
# ======================
y_pred_custom = (y_proba_log >= best_threshold).astype(int)
cm = confusion_matrix(y_test, y_pred_custom)

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title(f"Matrice de confusion (seuil={best_threshold:.2f})")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

# 9bis. Matrice de confusion + rapport aux seuils choisis (RF)
# ======================
# 1) Seuil F1-max (RF)
y_pred_rf_f1 = (y_proba_rf >= best_threshold_rf).astype(int)
cm_rf_f1 = confusion_matrix(y_test, y_pred_rf_f1)
sns.heatmap(cm_rf_f1, annot=True, fmt="d", cmap="Greens")
plt.title(f"Random Forest - Matrice de confusion (seuil F1-max={best_threshold_rf:.2f})")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

print("Classification Report - RF (seuil F1-max)")
print(classification_report(y_test, y_pred_rf_f1))

# 2) Option métier: maximiser le rappel sous contrainte de précision (ex. >= 0.60)
target_precision = 0.60  # ajuste selon ton budget/offres
best_rec, best_th_prec_constrained = -1, None
for t in thresholds_rf:
    y_pred_t = (y_proba_rf >= t).astype(int)
    p = precision_score(y_test, y_pred_t, zero_division=0)
    r = recall_score(y_test, y_pred_t)
    if p >= target_precision and r > best_rec:
        best_rec, best_th_prec_constrained = r, t

if best_th_prec_constrained is not None:
    print(f"Seuil RF (précision ≥ {target_precision:.2f}) = {best_th_prec_constrained:.2f} | Rappel = {best_rec:.3f}")
    y_pred_rf_precC = (y_proba_rf >= best_th_prec_constrained).astype(int)
    print("Classification Report - RF (seuil précision-contraint)")
    print(classification_report(y_test, y_pred_rf_precC))
    cm_rf_precC = confusion_matrix(y_test, y_pred_rf_precC)
    sns.heatmap(cm_rf_precC, annot=True, fmt="d", cmap="Oranges")
    plt.title(f"Random Forest - Matrice de confusion (seuil précision≥{target_precision:.0%} = {best_th_prec_constrained:.2f})")
    plt.xlabel("Prédit")
    plt.ylabel("Réel")
    plt.show()
else:
    print(f"Aucun seuil ne satisfait précision ≥ {target_precision:.2f}. Abaisse la contrainte ou essaie un autre objectif.")

# 3) Option Top-k (si tu as un budget: contacter p.ex. top 10% des plus à risque)
k_ratio = 0.10  # 10% des clients
k = max(1, int(k_ratio * len(y_proba_rf)))
t_k = np.partition(y_proba_rf, -k)[-k]  # seuil tel qu’~10% au-dessus
print(f"Seuil RF pour top-{int(k_ratio*100)}% ≈ {t_k:.3f} (à valider selon tes contraintes)")

# ======================
# 10. Analyse par état (où le churn est plus élevé ?)
# ======================
if "State" in df.columns:
    churn_by_state = df.groupby("State")["Churn"].apply(lambda x: (x==True).mean()).reset_index()
    churn_by_state.columns = ["State", "ChurnRate"]

    top10 = churn_by_state.sort_values("ChurnRate", ascending=False).head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(data=top10, x="State", y="ChurnRate")
    plt.title("Top 10 États avec le plus de churn")
    plt.show()

# ======================
# 11. Carte interactive des churners par état
# ======================
if "State" in df.columns:
    fig = px.choropleth(
        churn_by_state,
        locations="State",
        locationmode="USA-states",
        color="ChurnRate",
        scope="usa",
        color_continuous_scale="Reds",
        title="Taux de churn par État"
    )
    fig.show()


# ======================
# 12. Feature importance (Random Forest)
# ======================
importances = rf.feature_importances_
feat_imp = pd.Series(importances, index=X_train_scaled_df.columns).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feat_imp.values[:10], y=feat_imp.index[:10])
plt.title("Top 10 Feature Importances (Random Forest)")
plt.show()


# ======================
# 13. Top clients à risque (Logistic Regression)
# ======================
X_test_copy = X_test.copy()
X_test_copy["Churn_Proba"] = y_proba_log
top_risk = X_test_copy.sort_values("Churn_Proba", ascending=False).head(10)
top_risk
