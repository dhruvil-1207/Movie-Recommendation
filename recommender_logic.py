import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import streamlit as st

# We use st.cache_data so the CSVs only load once into memory
@st.cache_data
def load_data():
    movies = pd.read_csv('ml-latest-small/movies.csv')
    ratings = pd.read_csv('ml-latest-small/ratings.csv')
    title_to_id = dict(zip(movies['title'], movies['movieId']))
    return movies, ratings, title_to_id

# --- FAST APPROACH: COSINE SIMILARITY ---
@st.cache_data
def get_fast_recs(movie_title, _movies, _ratings):
    # Filter for popular movies to keep matrix small (as you did in your notebook)
    rating_counts = _ratings.groupby("movieId").size()
    popular_ids = rating_counts[rating_counts >= 50].index
    
    data_pop = _ratings[_ratings['movieId'].isin(popular_ids)]
    data_merged = pd.merge(data_pop, _movies[['movieId', 'title']], on='movieId')
    
    pivot = data_merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    sim_matrix = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)
    
    if movie_title in sim_df.columns:
        return sim_df[movie_title].sort_values(ascending=False).iloc[1:6]
    return None

# --- DEEP DIVE: SVD RETRAINING ---
def recommend_with_retrain(user_prefs, _movies, _ratings, _title_to_id):
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Create new user data
    new_user_id = 999
    new_rows = []
    for title, rating in user_prefs.items():
        if title in _title_to_id:
            new_rows.append({'userId': new_user_id, 'movieId': _title_to_id[title], 'rating': rating})
    
    new_ratings_df = pd.DataFrame(new_rows)
    combined_df = pd.concat([_ratings, new_ratings_df], ignore_index=True)
    
    # Load into Surprise
    data_surprise = Dataset.load_from_df(combined_df[['userId', 'movieId', 'rating']], reader)
    trainset = data_surprise.build_full_trainset()
    
    # Train SVD
    model = SVD()
    model.fit(trainset)
    
    # Predict for unwatched
    all_ids = _movies['movieId'].unique()
    rated_ids = [row['movieId'] for row in new_rows]
    unwatched = [m for m in all_ids if m not in rated_ids]
    
    preds = []
    for m_id in unwatched:
        preds.append((m_id, model.predict(new_user_id, m_id).est))
    
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:5]