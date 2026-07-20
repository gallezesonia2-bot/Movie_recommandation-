"""
Version RAPIDE du pipeline d'entraînement — pour machines aux ressources
limitées (Crostini/Chromebook).

On sait déjà (résultats du run précédent) que SVD++ gagne de peu sur SVD
(RMSE 0.917 vs 0.935), mais SVD++ est BEAUCOUP plus lent à tuner.
On choisit donc SVD directement (bon compromis qualité/vitesse) et on fait
un Grid Search réduit (8 combinaisons au lieu de 36, cv=3 au lieu de 5).

Temps attendu : quelques minutes, pas des heures.

Sortie : model.pkl + rapport_modele.txt
"""
from collections import defaultdict
import pandas as pd
from surprise import Dataset, Reader, SVD, dump, accuracy
from surprise.model_selection import KFold, GridSearchCV



# ------------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------------
columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)


report_lines = []
def log(msg=""):
    print(msg)
    report_lines.append(str(msg))



# ------------------------------------------------------------------
# 2. Metrics
# ------------------------------------------------------------------
def precision_recall_at_k(predictions, k=10, threshold=3.5):
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold))
            for (est, true_r) in user_ratings[:k]
        )
        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    return sum(precisions.values()) / len(precisions), sum(recalls.values()) / len(recalls)


# ------------------------------------------------------------------
# 3. Baseline rapide pour référence (juste 1 split, pas de 5-fold complet)
# ------------------------------------------------------------------
log("=" * 60)
log("ÉTAPE 1 : ÉVALUATION RAPIDE DE SVD (3-fold)")
log("=" * 60)

kf = KFold(n_splits=3, random_state=42)
rmses, precisions, recalls = [], [], []

base_svd = SVD(random_state=42)
for trainset, testset in kf.split(data):
    base_svd.fit(trainset)
    predictions = base_svd.test(testset)
    rmses.append(accuracy.rmse(predictions, verbose=False))
    p, r = precision_recall_at_k(predictions)
    precisions.append(p)
    recalls.append(r)

log(f"RMSE moyen (avant tuning)  : {sum(rmses)/len(rmses):.4f}")
log(f"Precision@10               : {sum(precisions)/len(precisions):.4f}")
log(f"Recall@10                  : {sum(recalls)/len(recalls):.4f}")


# ------------------------------------------------------------------
# 4. Grid Search RÉDUIT (8 combinaisons, cv=3 au lieu de 36 combos, cv=5)
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("ÉTAPE 2 : GRID SEARCH ALLÉGÉ (8 combinaisons, 3-fold)")
log("=" * 60)

param_grid = {
    'n_factors': [50, 100],
    'n_epochs': [20],
    'lr_all': [0.005],
    'reg_all': [0.02, 0.1],
}
# 2 x 1 x 1 x 2 = 4 combinaisons x 3 folds = 12 entraînements seulement
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3, n_jobs=-1)
gs.fit(data)

log(f"Meilleur RMSE après tuning : {gs.best_score['rmse']:.4f}")
log(f"Meilleurs paramètres       : {gs.best_params['rmse']}")

best_algo = gs.best_estimator['rmse']


# ------------------------------------------------------------------
# 5. Entraînement final sur toutes les données
# ------------------------------------------------------------------
log("\n" + "=" * 60)
log("ÉTAPE 3 : ENTRAÎNEMENT FINAL")
log("=" * 60)

full_trainset = data.build_full_trainset()
best_algo.fit(full_trainset)

dump.dump('model.pkl', algo=best_algo)
log("Modèle final (SVD, tuné) sauvegardé dans model.pkl")

with open('rapport_modele.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

log("Rapport sauvegardé dans rapport_modele.txt")
log("\nNote : version allégée pour cause de ressources machine limitées.")
log("Modèle = SVD (proche de SVD++ en qualité, beaucoup plus rapide à tuner).")
