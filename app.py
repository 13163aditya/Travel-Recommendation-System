import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")

FEATURES = [
    "Name_x",
    "State",
    "Type",
    "BestTimeToVisit",
    "Preferences",
    "Gender",
    "NumberOfAdults",
    "NumberOfChildren",
]

with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

destinations_df = pd.read_csv(os.path.join(DATA_DIR, "Expanded_Destinations.csv"))
userhistory_df = pd.read_csv(
    os.path.join(DATA_DIR, "Final_Updated_Expanded_UserHistory.csv")
)
df = pd.read_csv(os.path.join(DATA_DIR, "final_df.csv"))

# Build User-Item Matrix & Cosine Similarity Matrix
user_item_matrix = userhistory_df.pivot_table(
    index="UserID",
    columns="DestinationID",
    values="ExperienceRating",
    fill_value=0,
)
user_similarity = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(
    user_similarity,
    index=user_item_matrix.index,
    columns=user_item_matrix.index,
)


def hybrid_recommend(
    user_id=None,
    selected_name="",
    selected_type="",
    selected_state="",
    selected_time="",
    selected_pref="",
):
    """
    Supports leaving any field blank. Dynamically redistributes weights
    across whichever fields the user actually filled in.
    """
    candidates = destinations_df.copy()

    # 1. Collaborative Filtering Score (0.0 to 1.0)
    if user_id and user_id in user_similarity_df.index:
        similar_users = user_similarity_df.loc[user_id].drop(user_id)
        top_similar_user_ids = similar_users.sort_values(ascending=False).head(10).index
        cf_scores = user_item_matrix.loc[top_similar_user_ids].mean(axis=0)
        candidates["cf_score"] = candidates["DestinationID"].map(cf_scores).fillna(0)
        max_cf = candidates["cf_score"].max()
        if max_cf > 0:
            candidates["cf_score"] = candidates["cf_score"] / max_cf
    else:
        # Fallback if User ID is left blank or not found
        max_pop = candidates["Popularity"].max()
        candidates["cf_score"] = (
            candidates["Popularity"] / max_pop if max_pop > 0 else 0.5
        )

    # 2. Dynamic Content-Based Score (only evaluates non-empty inputs)
    active_weights = []
    active_matches = []

    if selected_name:
        active_weights.append(0.35)
        active_matches.append(
            np.where(candidates["Name"].str.lower() == selected_name.lower(), 1.0, 0.0)
        )
    if selected_type:
        active_weights.append(0.35)
        active_matches.append(
            np.where(candidates["Type"].str.lower() == selected_type.lower(), 1.0, 0.0)
        )
    if selected_state:
        active_weights.append(0.25)
        active_matches.append(
            np.where(candidates["State"].str.lower() == selected_state.lower(), 1.0, 0.0)
        )
    if selected_time:
        active_weights.append(0.15)
        active_matches.append(
            np.where(
                candidates["BestTimeToVisit"].str.lower() == selected_time.lower(),
                1.0,
                0.0,
            )
        )
    if selected_pref:
        # Match keywords inside Preferences (e.g. "Beaches, Historical") against Type
        pref_keywords = [p.strip().lower() for p in selected_pref.split(",")]
        pref_match = candidates["Type"].str.lower().apply(
            lambda t: 1.0 if any(k in t or t in k for k in pref_keywords) else 0.0
        )
        active_weights.append(0.20)
        active_matches.append(pref_match.values)

    # Normalize content score across only the fields the user provided
    if active_weights:
        total_w = sum(active_weights)
        content_score = np.zeros(len(candidates))
        for w, m in zip(active_weights, active_matches):
            content_score += (w / total_w) * m
        candidates["content_score"] = content_score
        # Blend 65% provided preferences + 35% collaborative filtering
        candidates["hybrid_score"] = (
            0.65 * candidates["content_score"] + 0.35 * candidates["cf_score"]
        )
    else:
        # If ALL preference dropdowns were left blank, rely 100% on Collaborative Filtering
        candidates["hybrid_score"] = candidates["cf_score"]

    candidates["MatchPercent"] = (candidates["hybrid_score"] * 100).round(0).astype(int)
    candidates["Popularity"] = candidates["Popularity"].round(2)

    ranked = candidates.sort_values(
        by=["hybrid_score", "Popularity"], ascending=[False, False]
    )
    unique_recommendations = ranked.drop_duplicates(subset=["Name"], keep="first").head(5)

    return unique_recommendations[
        [
            "DestinationID",
            "Name",
            "State",
            "Type",
            "Popularity",
            "BestTimeToVisit",
            "MatchPercent",
        ]
    ]


def predict_popularity(user_input):
    """Encodes user input safely and predicts destination popularity."""
    encoded_input = {}
    for feature in FEATURES:
        val = user_input[feature]
        if feature in label_encoders:
            encoder = label_encoders[feature]
            if val and val in encoder.classes_:
                encoded_input[feature] = encoder.transform([val])[0]
            else:
                # Default to most common class (index 0) when left blank
                encoded_input[feature] = 0
        else:
            encoded_input[feature] = val

    input_df = pd.DataFrame([encoded_input])[FEATURES]
    prediction = model.predict(input_df)[0]
    return round(float(prediction), 2)


def get_dropdown_options():
    return {
        "names": sorted(destinations_df["Name"].dropna().unique().tolist()),
        "states": sorted(destinations_df["State"].dropna().unique().tolist()),
        "types": sorted(destinations_df["Type"].dropna().unique().tolist()),
        "best_times": sorted(
            destinations_df["BestTimeToVisit"].dropna().unique().tolist()
        ),
        "preferences": (
            sorted(df["Preferences"].dropna().unique().tolist())
            if "Preferences" in df.columns
            else ["Adventure", "Historical", "Nature", "Relaxation"]
        ),
    }


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommendation")
def recommendation():
    options = get_dropdown_options()
    return render_template("recommendation.html", options=options, form_data={})


@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    options = get_dropdown_options()
    if request.method == "POST":
        try:
            raw_user_id = request.form.get("user_id", "").strip()
            user_id = int(raw_user_id) if raw_user_id else None

            raw_adults = request.form.get("adults", "").strip()
            adults = int(raw_adults) if raw_adults else 2

            raw_children = request.form.get("children", "").strip()
            children = int(raw_children) if raw_children else 0

            selected_name = request.form.get("name", "").strip()
            selected_state = request.form.get("state", "").strip()
            selected_type = request.form.get("type", "").strip()
            selected_time = request.form.get("best_time", "").strip()
            selected_pref = request.form.get("preferences", "").strip()
            selected_gender = request.form.get("gender", "").strip()

            recommended_destinations = hybrid_recommend(
                user_id=user_id,
                selected_name=selected_name,
                selected_type=selected_type,
                selected_state=selected_state,
                selected_time=selected_time,
                selected_pref=selected_pref,
            )

            # If Destination Name was left blank, use the #1 recommended destination for ML prediction
            top_row = recommended_destinations.iloc[0]
            user_input = {
                "Name_x": selected_name or top_row["Name"],
                "Type": selected_type or top_row["Type"],
                "State": selected_state or top_row["State"],
                "BestTimeToVisit": selected_time or top_row["BestTimeToVisit"],
                "Preferences": selected_pref or "Adventure",
                "Gender": selected_gender or "Male",
                "NumberOfAdults": adults,
                "NumberOfChildren": children,
            }

            predicted_popularity = predict_popularity(user_input)

            form_data = {
                "user_id": raw_user_id,
                "name": selected_name,
                "state": selected_state,
                "type": selected_type,
                "best_time": selected_time,
                "preferences": selected_pref,
                "gender": selected_gender,
                "adults": raw_adults,
                "children": raw_children,
            }

            return render_template(
                "recommendation.html",
                options=options,
                form_data=form_data,
                recommended_destinations=recommended_destinations,
                predicted_popularity=predicted_popularity,
                submitted=True,
            )
        except Exception as e:
            return render_template(
                "recommendation.html",
                options=options,
                form_data={},
                error=f"Error generating recommendation: {str(e)}",
            )

    return render_template("recommendation.html", options=options, form_data={})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)