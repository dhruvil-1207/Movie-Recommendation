import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
import streamlit as st
import os

# Get the directory where this script lives (points to project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# We use st.cache_data so the CSVs only load once into memory
@st.cache_data
def load_data():
    # Paths are now relative to the project root
    movies_path = os.path.join(BASE_DIR, 'data', 'ml-latest-small', 'movies.csv')
    ratings_path = os.path.join(BASE_DIR, 'data', 'ml-latest-small', 'ratings.csv')
    
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Could not find movies.csv at {movies_path}")
        
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(ratings_path)
    
    # Pre-calculate a mapping for the SVD logic
    title_to_id = pd.Series(movies.movieId.values, index=movies.title).to_dict()
    id_to_title = pd.Series(movies.title.values, index=movies.movieId).to_dict()
    
    return movies, ratings, title_to_id, id_to_title

# --- APPROACH 1: COSINE SIMILARITY (Item-Item) ---
@st.cache_data
def get_fast_recs(movie_title, _movies, _ratings):
    # Filter for popular movies (at least 50 ratings) to keep matrix manageable
    rating_counts = _ratings.groupby("movieId").size()
    popular_ids = rating_counts[rating_counts >= 50].index
    
    data_pop = _ratings[_ratings['movieId'].isin(popular_ids)]
    data_merged = pd.merge(data_pop, _movies[['movieId', 'title']], on='movieId')
    
    # Create the Pivot Table
    pivot = data_merged.pivot_table(index='userId', columns='title', values='rating').fillna(0)
    
    # Calculate Similarity
    sim_matrix = cosine_similarity(pivot.T)
    sim_df = pd.DataFrame(sim_matrix, index=pivot.columns, columns=pivot.columns)
    
    if movie_title in sim_df.columns:
        # Return top 5 similar movies (excluding itself)
        return sim_df[movie_title].sort_values(ascending=False).iloc[1:6]
    return None

# --- APPROACH 2: SVD (Collaborative Filtering) ---
def recommend_with_retrain(user_prefs, _movies, _ratings, _title_to_id, _id_to_title):
    """
    user_prefs: Dict like {"Toy Story (1995)": 5.0, "Jumanji (1995)": 3.0}
    """
    reader = Reader(rating_scale=(0.5, 5.0))
    
    # Create new user entry
    new_user_id = 999
    new_rows = []
    for title, rating in user_prefs.items():
        if title in _title_to_id:
            new_rows.append({'userId': new_user_id, 'movieId': _title_to_id[title], 'rating': rating})
    
    if not new_rows:
        return []

    new_ratings_df = pd.DataFrame(new_rows)
    # Combine with existing ratings for training
    combined_df = pd.concat([_ratings[['userId', 'movieId', 'rating']], new_ratings_df], ignore_index=True)
    
    # Load into Surprise and Train
    data_surprise = Dataset.load_from_df(combined_df, reader)
    trainset = data_surprise.build_full_trainset()
    
    model = SVD()
    model.fit(trainset)
    
    # Predict for all movies the user hasn't rated yet
    all_movie_ids = _movies['movieId'].unique()
    rated_movie_ids = [row['movieId'] for row in new_rows]
    
    preds = []
    for m_id in all_movie_ids:
        if m_id not in rated_movie_ids:
            est_rating = model.predict(new_user_id, m_id).est
            preds.append((_id_to_title.get(m_id, "Unknown"), est_rating))
    
    # Sort by predicted rating
    preds.sort(key=lambda x: x[1], reverse=True)
    return preds[:5]