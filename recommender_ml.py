# recommender_ml.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

# ---------- Load data ----------
DF_PATH = "Destinations.csv"
df = pd.read_csv(DF_PATH)

# Normalize some fields
df['Type'] = df['Type'].fillna('').astype(str)
df['Popular_Attractions'] = df['Popular_Attractions'].fillna('').astype(str)
df['Country'] = df['Country'].fillna('').astype(str)
df['Best_Season'] = df['Best_Season'].fillna('').astype(str)
df['Average_Cost'] = pd.to_numeric(df['Average_Cost'], errors='coerce').fillna(0)
df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce').fillna(0)

# ---------- Text preprocessing ----------
def clean_text(s):
    s = str(s).lower()
    s = re.sub(r'[^a-z0-9\s\|,-]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

df['combined_text'] = (df['Type'] + " " + df['Popular_Attractions'] + " " + df['Country']).apply(clean_text)

# ---------- Build TF-IDF ----------
vectorizer = TfidfVectorizer(ngram_range=(1,2), stop_words='english', max_features=5000)
tfidf_matrix = vectorizer.fit_transform(df['combined_text'])

# Precompute cosine similarity between destinations (useful for "similar to X" queries)
dest_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ---------- Recommendation functions ----------
def month_match_score(best_season_string, user_month):
    # crude month partial match: checks if user_month short form appears in best_season
    try:
        if pd.isna(best_season_string) or best_season_string == '':
            return 0.5
        bs = best_season_string.lower()
        um = user_month[:3].lower()
        return 1.0 if um in bs else 0.8
    except:
        return 0.8

def recommend_by_preferences(preferred_types=None, budget=None, month=None, top_k=5):
    """
    preferred_types: list of keywords, e.g. ['beach','adventure']
    budget: numeric (USD per day) -> will penalize expensive places
    month: string like 'April' or 'Dec'
    """
    if preferred_types is None:
        preferred_types = []
    # Build user query text
    query_text = " ".join(preferred_types).lower()
    query_text = clean_text(query_text)
    query_vec = vectorizer.transform([query_text])
    sim_scores = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # create scoring dataframe
    scores_df = df.copy()
    scores_df['sim_score'] = sim_scores

    # Season adjustment
    if month:
        scores_df['season_factor'] = scores_df['Best_Season'].apply(lambda x: month_match_score(x, month))
    else:
        scores_df['season_factor'] = 1.0

    # Budget penalty (soft)
    if budget is not None:
        try:
            b = float(budget)
            scores_df['budget_penalty'] = (scores_df['Average_Cost'] - b) / (b + 1)
            # clip penalty so it doesn't explode
            scores_df['budget_penalty'] = scores_df['budget_penalty'].clip(-1, 2)
        except:
            scores_df['budget_penalty'] = 0.0
    else:
        scores_df['budget_penalty'] = 0.0

    # Combine into final score
    # weights: sim_score (0.6), rating (normalized -> 0.2), season_factor (0.1), budget penalty (0.1)
    # normalize rating to 0-1 (assuming rating 0-5)
    scores_df['rating_norm'] = scores_df['Rating'] / 5.0
    scores_df['final_score'] = (
        0.6 * scores_df['sim_score'] +
        0.2 * scores_df['rating_norm'] +
        0.1 * scores_df['season_factor'] -
        0.1 * scores_df['budget_penalty']
    )

    results = scores_df.sort_values('final_score', ascending=False).head(top_k)
    return results[['id','Destination','Country','Type','Average_Cost','Best_Season','Rating','final_score']].reset_index(drop=True)

def recommend_similar_to(destination_name, top_k=5):
    # find index
    idx = df[df['Destination'].str.lower() == destination_name.lower()].index
    if len(idx) == 0:
        return pd.DataFrame()  # no such destination
    idx = idx[0]
    sim_scores = list(enumerate(dest_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    top_indices = [i for i, s in sim_scores[1: top_k+1]]  # skip self
    return df.iloc[top_indices][['id','Destination','Country','Type','Average_Cost','Best_Season','Rating']].reset_index(drop=True)

# ---------- Quick test ----------
if __name__ == "__main__":
    print("enter your preferred type:")
    print("=== Sample: recommend for ['beach'] ")
    print(recommend_by_preferences(['beach'],top_k=2).to_string(index=False))
    print("\n=== Sample: similar to 'Bali' ===")
    print(recommend_similar_to('bangalore',top_k=1).to_string(index=False))

