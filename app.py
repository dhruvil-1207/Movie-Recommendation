import streamlit as st
from recommender_logic import load_data, get_fast_recs, recommend_with_retrain

st.set_page_config(page_title="Movie Matcher AI", layout="wide")

# Load data once
movies, ratings, title_to_id = load_data()

st.title("🎬 Movie Recommendation System")
st.markdown("Enter multiple favorites to get highly personalized suggestions.")

# --- SIDEBAR: MULTI-MOVIE INPUT ---
st.sidebar.header("Your Preferences")

selected_titles = st.sidebar.multiselect(
    "Pick movies you love:", 
    options=movies['title'].values,
    default=["GoldenEye (1995)"]
)

user_prefs = {}
if selected_titles:
    st.sidebar.write("Rate your selections:")
    for title in selected_titles:
        # ADDED 'step=0.5' to fix the increments
        score = st.sidebar.slider(
            f"{title}", 
            min_value=0.5, 
            max_value=5.0, 
            value=4.5, 
            step=0.5, 
            key=f"sr_{title}"
        )
        user_prefs[title] = score
else:
    st.sidebar.warning("Please select at least one movie.")

# --- MAIN INTERFACE (Tabs) ---
tab1, tab2 = st.tabs(["⚡ Fast Match (Similarity)", "🧠 AI Deep Dive (SVD)"])

with tab1:
    st.subheader("Similar to your first pick")
    if selected_titles:
        primary_movie = selected_titles[0]
        if st.button("Find Similar to " + primary_movie):
            results = get_fast_recs(primary_movie, movies, ratings)
            if results is not None:
                for title, score in results.items():
                    st.write(f"**{title}** (Similarity Score: {score:.2f})")
            else:
                st.error("Movie patterns not found.")
    else:
        st.write("Select a movie in the sidebar first!")

with tab2:
    st.subheader("Personalized AI Predictions")
    st.info("This model analyzes your entire list to predict what else you'll love.")
    
    if st.button("Run Deep Dive"):
        if user_prefs:
            with st.spinner("Training SVD Model on 100k ratings..."):
                predictions = recommend_with_retrain(user_prefs, movies, ratings, title_to_id)
                
                for m_id, score in predictions:
                    name = movies[movies['movieId'] == m_id]['title'].values[0]
                    st.write(f"✅ **{name}** - Predicted Rating: {score:.2f}★")
        else:
            st.error("Add some movies in the sidebar to begin.")

st.markdown("---")
st.caption("Built for BVM Engineering Project. Powered by Scikit-Surprise & Streamlit.")