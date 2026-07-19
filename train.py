"""
Pipeline ML complet pour le système de recommandation de films.

Étapes :
  1. Comparaison élargie d'algorithmes (collaborative filtering pur)
  2. Grid Search sur le meilleur candidat
  3. Évaluation approfondie : RMSE, Precision@K, Recall@K, Diversité, Coverage
  4. Sauvegarde du modèle final + rapport texte pour le client

Note : ce script fait du collaborative filtering PUR (basé uniquement sur les
notes). L'enrichissement par genres n'est utilisé que côté API pour le
cold-start (utilisateurs sans historique) et l'affichage par catégorie —
ce n'est pas un modèle hybride entraîné sur les genres.

Sortie : model.pkl + rapport_modele.txt
"""
from collections import defaultdict

import numpy as np
import pandas as pd
from surprise import (
    Dataset, Reader, SVD, SVDpp, KNNWithMeans, NMF,
    BaselineOnly, SlopeOne, CoClustering, dump, accuracy
)
from surprise.model_selection import KFold, GridSearchCV


# ------------------------------------------------------------------
# 1. Chargement des données
# ------------------------------------------------------------------
columns = ['user_id', 'item_id', 'rating', 'timestamp']
ratings = pd.read_csv('ml-100k/u.data', sep='\t', names=columns)

genre_cols = [
    'unknown', 'Action', 'Adventure', 'Animation', "Children's",
    'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
    'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance',
    'Sci-Fi', 'Thriller', 'War', 'Western'
]
item_cols = ['item_id', 'title', 'release_date', 'video_release_date', 'imdb_url'] + genre_cols
movies = pd.read_csv('ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')

reader = Reader(rating_scale=(1, 5))
data = Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']], reader)

report_lines = []
def log(msg=""):
    """Affiche à l'écran ET garde une trace pour le rapport final (preuve pour le client)."""
    print(msg)
    report_lines.append(str(msg))


# ------------------------------------------------------------------
# 2. Métriques : Precision@K / Recall@K
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

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    return avg_precision, avg_recall


# ------------------------------------------------------------------
# 3. Métriques : Diversité & Coverage
#    (un modèle peut avoir un bon RMSE mais recommander toujours les
#    mêmes 20 films populaires -> mauvaise expérience utilisateur réelle)
# ------------------------------------------------------------------
def diversity_and_coverage(algo, trainset, all_item_ids, n=10, sample_users=200):
    """
    Coverage  = % du catalogue total qui apparaît au moins une fois
                dans les recos de tous les users échantillonnés
    Diversité = proportion de films UNIQUES parmi toutes les recos données
                (proche de 1 = le modèle varie ses recos d'un user à l'autre,
                 proche de 0 = il recommande presque toujours les mêmes films)
    """
    inner_uids = trainset.all_users()
    sampled = np.random.choice(inner_uids, min(sample_users, len(inner_uids)), replace=False)

    recommended_items = set()
    all_recommendations = []  # avec doublons, pour calculer la diversité

    for inner_uid in sampled:
        raw_uid = trainset.to_raw_uid(inner_uid)
        rated = {j for (j, _) in trainset.ur[inner_uid]}
        rated_raw = {trainset.to_raw_iid(j) for j in rated}
        candidates = [iid for iid in all_item_ids if iid not in rated_raw]

        preds = [algo.predict(raw_uid, iid) for iid in candidates]
        preds.sort(key=lambda p: p.est, reverse=True)
        top_n = [p.iid for p in preds[:n]]

        recommended_items.update(top_n)
        all_recommendations.extend(top_n)

    coverage = len(recommended_items) / len(all_item_ids)
    diversity = len(set(all_recommendations)) / len(all_recommendations) if all_recommendations else 0
    return diversity, coverage


# ------------------------------------------------------------------
# 4. Comparaison élargie d'algorithmes
# ------------------------------------------------------------------
algorithms = {
    'Baseline (référence)': BaselineOnly(),
    'SVD': SVD(random_state=42),
    'SVD++': SVDpp(random_state=42),
    'KNNWithMeans': KNNWithMeans(k=40, sim_options={'name': 'cosine', 'user_based': True}, verbose=False),
    'NMF': NMF(random_state=42),
    'SlopeOne': SlopeOne(),
    'CoClustering': CoClustering(random_state=42),
}

log("=" * 65)
log("ÉTAPE 1 : COMPARAISON DE 7 ALGORITHMES (5-fold cross-validation)")
log("=" * 65)

results_summary = []
kf = KFold(n_splits=5, random_state=42)
all_item_ids = movies['item_id'].tolist()

for name, algo in algorithms.items():
    log(f"\n--- {name} ---")
    rmses, precisions, recalls = [], [], []
    last_trainset = None

    for trainset, testset in kf.split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)

        rmse = accuracy.rmse(predictions, verbose=False)
        prec, rec = precision_recall_at_k(predictions, k=10, threshold=3.5)

        rmses.append(rmse)
        precisions.append(prec)
        recalls.append(rec)
        last_trainset = trainset

    avg_rmse = sum(rmses) / len(rmses)
    avg_prec = sum(precisions) / len(precisions)
    avg_rec = sum(recalls) / len(recalls)
    diversity, coverage = diversity_and_coverage(algo, last_trainset, all_item_ids)

    log(f"RMSE moyen           : {avg_rmse:.4f}")
    log(f"Precision@10 moyenne : {avg_prec:.4f}")
    log(f"Recall@10 moyen      : {avg_rec:.4f}")
    log(f"Diversité            : {diversity:.4f}")
    log(f"Coverage catalogue   : {coverage:.2%}")

    results_summary.append({
        'model': name, 'rmse': avg_rmse, 'precision@10': avg_prec,
        'recall@10': avg_rec, 'diversity': diversity, 'coverage': coverage
    })

summary_df = pd.DataFrame(results_summary).sort_values('rmse')
log("\n" + "=" * 65)
log("RÉSUMÉ COMPARATIF")
log("=" * 65)
log(summary_df.to_string(index=False))

best_name = summary_df.iloc[0]['model']
log(f"\n>>> Meilleur modèle (RMSE le plus bas) : {best_name}")


# ------------------------------------------------------------------
# 5. Grid Search sur le meilleur modèle (si SVD ou SVD++, sinon on garde tel quel)
# ------------------------------------------------------------------
log("\n" + "=" * 65)
log("ÉTAPE 2 : GRID SEARCH (optimisation fine des hyperparamètres)")
log("=" * 65)

if best_name in ('SVD', 'SVD++'):
    AlgoClass = SVD if best_name == 'SVD' else SVDpp
    param_grid = {
        'n_factors': [50, 100, 150],
        'n_epochs': [20, 30],
        'lr_all': [0.002, 0.005, 0.01],
        'reg_all': [0.02, 0.1, 0.4],
    }
    gs = GridSearchCV(AlgoClass, param_grid, measures=['rmse'], cv=5, n_jobs=-1)
    gs.fit(data)

    log(f"Meilleur RMSE après tuning : {gs.best_score['rmse']:.4f}")
    log(f"Meilleurs paramètres       : {gs.best_params['rmse']}")

    best_algo = gs.best_estimator['rmse']
else:
    log(f"Grid search non applicable directement pour {best_name}, "
        f"on garde la configuration par défaut.")
    best_algo = algorithms[best_name]


# ------------------------------------------------------------------
# 6. Ré-entraînement final sur TOUTES les données
# ------------------------------------------------------------------
log("\n" + "=" * 65)
log("ÉTAPE 3 : ENTRAÎNEMENT FINAL SUR L'INTÉGRALITÉ DES DONNÉES")
log("=" * 65)

full_trainset = data.build_full_trainset()
best_algo.fit(full_trainset)

dump.dump('model.pkl', algo=best_algo)
log(f"Modèle final ({best_name}, tuné) sauvegardé dans model.pkl")


# ------------------------------------------------------------------
# 7. Rapport texte pour justifier le choix au client
# ------------------------------------------------------------------
with open('rapport_modele.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(report_lines))

log("\nRapport complet sauvegardé dans rapport_modele.txt (à montrer au client)")