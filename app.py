import streamlit as st
from src.recommender_logic import load_data, get_fast_recs, recommend_with_retrain

st.set_page_config(page_title="Movie Matcher AI", layout="wide", page_icon="🎬")

# Load data once - Unpacking 4 values as per updated logic file
movies, ratings, title_to_id, id_to_title = load_data()

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
        if st.button("Find Similar to " + primary_movie, key="btn_fast"):
            with st.spinner("Calculating similarity..."):
                results = get_fast_recs(primary_movie, movies, ratings)
                
                if results is not None:
                    # Displaying results in a nice format
                    for title, score in results.items():
                        st.write(f"⭐ **{title}** (Similarity Score: {score:.2f})")
                else:
                    st.error("This movie doesn't have enough ratings to find patterns. Try a more popular one!")
    else:
        st.info("Select a movie in the sidebar first!")

with tab2:
    st.subheader("Personalized AI Predictions")
    st.info("This model retrains SVD on 100k ratings plus your input to predict what you'll love.")
    
    if st.button("Run Deep Dive", key="btn_svd"):
        if user_prefs:
            with st.spinner("Retraining SVD Model... This takes a few seconds on your i7."):
                # Passing all necessary mappings to the logic function
                predictions = recommend_with_retrain(
                    user_prefs, movies, ratings, title_to_id, id_to_title
                )
                
                if predictions:
                    st.success("Top 5 Predicted Matches for You:")
                    for name, score in predictions:
                        st.write(f"✅ **{name}** — Predicted Rating: {score:.2f} ★")
                else:
                    st.error("Something went wrong with the retraining logic.")
        else:
            st.error("Add some movies in the sidebar to begin.")

st.markdown("---")
