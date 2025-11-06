# streamlit_app.py
import streamlit as st
from recommender_ml import recommend_by_preferences, recommend_similar_to, df

st.markdown("""
    <style>
    /* 🌈 Background Gradient */
    .stApp {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        font-family: 'Poppins', sans-serif;
    }

    /* ✨ Headings */
    h1 {
        color: #1a237e;
        text-align: center;
        font-weight: 700;
        letter-spacing: 1px;
        text-shadow: 1px 1px 3px rgba(63, 81, 181, 0.3);
        margin-bottom: 0.2em;
    }

    h2, h3 {
        color: #283593;
        font-weight: 600;
    }

    /* 🌴 Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(135deg, #a8edea, #fed6e3);
        color: #000;
        font-weight: 500;
        border-right: 2px solid #fbc02d;
        box-shadow: 2px 0 8px rgba(0,0,0,0.1);
    }

    /* ✍️ Sidebar labels */
    .stTextInput label, .stNumberInput label, .stSlider label, .stSelectbox label {
        color: #5d4037;
        font-weight: bold;
    }

    /* 🌟 Button styling */
    div.stButton > button {
        background: skyblue;
        color: white;
        border: none;
        border-radius: 12px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: 600;
        box-shadow: 0 4px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:hover {
        transform: white;
        background: linear-gradient(90deg, #ff9966, #ff5e62);
        box-shadow: 0 4px 14px rgba(0,0,0,0.3);
    }

    /* 🧭 Glassy recommendation cards */
    .recommend-card {
        background: rgba(255, 255, 255, 0.65);
        backdrop-filter: blur(12px);
        padding: 22px;
        border-radius: 20px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 20px;
        border-left: 6px solid #42a5f5;
        transition: all 0.3s ease;
    }
    .recommend-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(0,0,0,0.15);
    }

    /* ✨ Horizontal rule */
    hr {
        border: none;
        height: 2px;
        background-color: #90caf9;
        margin: 25px 0;
    }

    /* 📦 Info box */
    .stAlert {
        border-radius: 10px !important;
    }

    /* Text */
    p {
        font-size: 15px;
        color: #263238;
    }
    </style>
""", unsafe_allow_html=True)
st.set_page_config(page_title="Travel Recommender", page_icon="🌍")
st.title("🌍 Intelligent Travel Recommendation System")
st.write("Provide your preferences and get recommended travel destinations.")

# Left column: input
st.sidebar.header("Your Preferences")
pref_input = st.sidebar.text_input("Preferred types (comma separated)", "beach,relax")
budget = st.sidebar.number_input("Budget per day (USD)", min_value=0, value=1000)
month = st.sidebar.text_input("Preferred month (e.g., April) (optional)", "")
top_k = st.sidebar.slider("How many results?", min_value=1, max_value=5

, value=5)

st.sidebar.markdown("---")
st.sidebar.write("Or pick an example destination to find similar places")
destination_choice = st.sidebar.selectbox("Similar to (optional):", [""] + df['Destination'].tolist())

# Main: compute recommendations
if st.sidebar.button("Get Recommendations"):
    if destination_choice:
        results = recommend_similar_to(destination_choice, top_k=top_k)
    else:
        preferred_types = [t.strip() for t in pref_input.split(",") if t.strip()]
        results = recommend_by_preferences(preferred_types, budget=budget, month=month, top_k=top_k)

    if results is None or results.empty:
        st.warning("No recommendations found. Try changing your filters.")
    else:
        for idx, row in results.iterrows():
            st.markdown(f"### {row['Destination']} — {row['Country']}")
            st.write(f"**Type:** {row['Type']} • **Avg cost/day:** ${row['Average_Cost']} • **Best season:** {row['Best_Season']} • **Rating:** {row['Rating']}")
            st.write(f"**Score:** {row.get('final_score', 'N/A'):.3f}" if 'final_score' in row else "")
            st.markdown("---")

st.info("Tip: increase dataset size and tune the scoring weights for better results.")
