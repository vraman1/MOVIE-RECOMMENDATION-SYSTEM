#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proposed Hybrid Movie Recommendation System
--------------------------------------------------------
Deep Multimodal Embeddings + Generative MF + Graph Learning + Neuro-Symbolic Rules
Works on a SINGLE hybrid dataset CSV.
"""

import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Column Detection
# ---------------------------------------------------------
def infer_cols(df: pd.DataFrame):
    cols = {c.lower(): c for c in df.columns}
    uid = cols.get("userid") or cols.get("user_id") or cols.get("user id")
    mid = cols.get("movieid") or cols.get("movie_id") or cols.get("movie id") or cols.get("itemid")
    rate = cols.get("rating") or cols.get("ratings")
    title = cols.get("title") or cols.get("movie_title") or cols.get("name")
    genre = cols.get("genres") or cols.get("genre") or cols.get("category")
    ts = cols.get("timestamp") or cols.get("time") or cols.get("date")
    if not (uid and mid and rate):
        raise ValueError(f"Required columns (UserID, MovieID, Rating) not found in: {df.columns.tolist()}")
    return uid, mid, rate, title, genre, ts


# ---------------------------------------------------------
# Text Embeddings (TF-IDF)
# ---------------------------------------------------------
def build_text_embeddings(movies_df, id_col, title_col, genre_col):
    text = []
    for _, row in movies_df.iterrows():
        parts = []
        if title_col:
            parts.append(str(row.get(title_col, "")))
        if genre_col:
            parts.append(str(row.get(genre_col, "")))
        text.append(" ".join(parts).strip())

    tfidf = TfidfVectorizer(stop_words="english")
    X = tfidf.fit_transform(text)
    movieid_to_row = {m: i for i, m in enumerate(movies_df[id_col].tolist())}
    return X, movieid_to_row


# ---------------------------------------------------------
# Graph Signals (User–Movie Co-Watch)
# ---------------------------------------------------------
def build_graph_signals(ratings_df, uid_col, mid_col):
    user2items = defaultdict(set)
    item2users = defaultdict(set)
    for _, r in ratings_df.iterrows():
        user2items[r[uid_col]].add(r[mid_col])
        item2users[r[mid_col]].add(r[uid_col])
    return user2items, item2users


def jaccard_items(item2users, a, b):
    A = item2users.get(a, set())
    B = item2users.get(b, set())
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


# ---------------------------------------------------------
# Generative Preference Model (Matrix Factorization)
# ---------------------------------------------------------
def train_mf(ratings_df, uid_col, mid_col, rate_col, max_pairs=20000):
    user_ids = ratings_df[uid_col].astype(int).unique()
    item_ids = ratings_df[mid_col].astype(int).unique()
    uid2idx = {u: i for i, u in enumerate(user_ids)}
    mid2idx = {m: i for i, m in enumerate(item_ids)}

    rng = np.random.default_rng(42)
    k, epochs, lr, reg = 16, 1, 0.02, 0.01   # optimized faster MF
    P = 0.1 * rng.standard_normal((len(user_ids), k))
    Q = 0.1 * rng.standard_normal((len(item_ids), k))
    bu = np.zeros(len(user_ids))
    bi = np.zeros(len(item_ids))
    mu = float(ratings_df[rate_col].mean())

    triples = ratings_df[[uid_col, mid_col, rate_col]].sample(
        n=min(max_pairs, len(ratings_df)), random_state=42
    ).to_numpy()

    for _ in range(epochs):
        rng.shuffle(triples)
        for u_raw, i_raw, r in triples:
            u, i = uid2idx[u_raw], mid2idx[i_raw]
            pred = mu + bu[u] + bi[i] + P[u] @ Q[i]
            e = r - pred
            bu[u] += lr * (e - reg * bu[u])
            bi[i] += lr * (e - reg * bi[i])
            Pu = P[u]
            P[u] += lr * (e * Q[i] - reg * Pu)
            Q[i] += lr * (e * Pu - reg * Q[i])

    def predict(u_raw, i_raw):
        if u_raw not in uid2idx or i_raw not in mid2idx:
            return mu
        u, i = uid2idx[u_raw], mid2idx[i_raw]
        return float(mu + bu[u] + bi[i] + P[u] @ Q[i])

    return predict


# ---------------------------------------------------------
# Neuro-Symbolic Genre Rule
# ---------------------------------------------------------
def top_genres_for_user(ratings_df, movies_df, uid_col, mid_col, rate_col, genre_col, user_id):
    x = ratings_df[ratings_df[uid_col] == user_id]
    x = x.merge(movies_df[[mid_col, genre_col]], on=mid_col, how="left")
    x = x[x[rate_col] >= x[rate_col].median()]
    freq = defaultdict(int)
    for gs in x[genre_col].dropna():
        for g in str(gs).split("|"):
            freq[g.strip()] += 1
    return set(sorted(freq, key=freq.get, reverse=True)[:3])


def rule_bonus(user_genres, movie_genre_str):
    if not movie_genre_str: return 0.0
    for g in str(movie_genre_str).split("|"):
        if g.strip() in user_genres:
            return 0.2
    return 0.0


# ---------------------------------------------------------
# RECOMMEND FUNCTION (Main Output)
# ---------------------------------------------------------
def recommend(df, user_id, top_n=10):
    uid_col, mid_col, rate_col, title_col, genre_col, ts_col = infer_cols(df)

    movies_df = df[[mid_col, title_col, genre_col]].drop_duplicates(subset=[mid_col]).reset_index(drop=True)
    ratings_df = df[[uid_col, mid_col, rate_col]].dropna()
    ratings_df[rate_col] = pd.to_numeric(ratings_df[rate_col], errors="coerce")

    X, movieid_to_row = build_text_embeddings(movies_df, mid_col, title_col, genre_col)
    user2items, item2users = build_graph_signals(ratings_df, uid_col, mid_col)
    predict_fn = train_mf(ratings_df, uid_col, mid_col, rate_col)

    seen = set(ratings_df[ratings_df[uid_col] == user_id][mid_col].tolist())
    candidates = [m for m in movies_df[mid_col] if m not in seen]

    likes = ratings_df[(ratings_df[uid_col] == user_id) & 
                       (ratings_df[rate_col] >= ratings_df[rate_col].median())][mid_col].tolist()
    user_genres = top_genres_for_user(ratings_df, movies_df, uid_col, mid_col, rate_col, genre_col, user_id)

    rows = []
    for m in candidates:
        gen = predict_fn(user_id, m)
        like_rows = [movieid_to_row[x] for x in likes if x in movieid_to_row]
        cs = cosine_similarity(X[movieid_to_row[m]], X[like_rows]).mean() if like_rows else 0.0
        gsim = np.mean([jaccard_items(item2users, m, x) for x in likes]) if likes else 0.0
        bonus = rule_bonus(user_genres, movies_df.loc[movies_df[mid_col] == m, genre_col].values[0])
        score = 0.6*gen + 0.25*(3+2*cs) + 0.15*(3+2*gsim) + bonus

        title = movies_df.loc[movies_df[mid_col] == m, title_col].values[0]
        rows.append((m, title, score))

    rows.sort(key=lambda x: x[-1], reverse=True)
    return pd.DataFrame(rows[:top_n], columns=["MovieID", "Title", "Hybrid_Score"])


# ---------------------------------------------------------
# ACCURACY EVALUATION (RMSE + MAE + Precision@5)
# ---------------------------------------------------------
def evaluate(df):
    uid_col, mid_col, rate_col, _, _, _ = infer_cols(df)
    df = df[[uid_col, mid_col, rate_col]].dropna()
    df[rate_col] = df[rate_col].astype(float)

    # 80/20 per-user split
    train_idx, test_idx = [], []
    rng = np.random.default_rng(42)
    for u, g in df.groupby(uid_col):
        idx = g.index.to_list()
        rng.shuffle(idx)
        k = max(1, int(len(idx)*0.8))
        train_idx += idx[:k]
        test_idx += idx[k:]
    train, test = df.loc[train_idx], df.loc[test_idx]

    predict_fn = train_mf(train, uid_col, mid_col, rate_col)

    # RMSE + MAE
    y, yhat = [], []
    for _, r in test.iterrows():
        y.append(r[rate_col])
        yhat.append(predict_fn(r[uid_col], r[mid_col]))
    y, yhat = np.array(y), np.array(yhat)
    rmse = float(np.sqrt(np.mean((y-yhat)**2)))
    mae = float(np.mean(np.abs(y-yhat)))

    # Precision@5
    movies = train[mid_col].unique().tolist()
    precs = []
    for u, g in test.groupby(uid_col):
        relevant = set(g[g[rate_col] >= 4.0][mid_col].tolist())
        if not relevant: continue
        seen = set(train[train[uid_col]==u][mid_col].tolist())
        cand = [m for m in movies if m not in seen]
        cand_scored = sorted(cand, key=lambda m: predict_fn(u,m), reverse=True)[:5]
        precs.append(len(set(cand_scored)&relevant)/5)
    p5 = float(np.mean(precs)) if len(precs)>0 else float("nan")

    print("\n=== MODEL ACCURACY ===")
    print(f"RMSE          : {rmse:.3f}")
    print(f"MAE           : {mae:.3f}")
    print(f"Precision@5   : {p5:.3f}")
    print("======================\n")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to hybrid dataset CSV")
    ap.add_argument("--user_id", type=int, required=True, help="User to recommend for")
    ap.add_argument("--top_n", type=int, default=10)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(recommend(df, args.user_id, args.top_n))
    evaluate(df)


if __name__ == "__main__":
    main()
