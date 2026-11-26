# Travel Recommendation System

A Flask web application that recommends travel destinations and predicts destination popularity using a hybrid recommendation approach and a `RandomForestRegressor` model.

- **Live Demo:** [https://travel-recommendation-system-v2fj.onrender.com/](https://travel-recommendation-system-v2fj.onrender.com/)

## Features

- **Hybrid Destination Recommendations:** Blends collaborative filtering (user–item cosine similarity on experience ratings, with a popularity fallback when `UserID` is omitted) and content-based matching across destination name, type, state, best time to visit, and preferences.
- **Dynamic Weight Redistribution:** Supports leaving preference fields blank and renormalizes weights across whichever inputs are provided.
- **Popularity Prediction:** Predicts destination popularity scores based on trip and traveler attributes (`Name_x`, `State`, `Type`, `BestTimeToVisit`, `Preferences`, `Gender`, `NumberOfAdults`, and `NumberOfChildren`).
- **Web Interface:** Served via Flask routes (`/`, `/recommendation`, and `/recommend`).

## Project Structure

- `app.py` — Flask application, hybrid recommendation engine, and popularity prediction inference.
- `train.py` — Trains the `RandomForestRegressor` model and saves `model.pkl` and `label_encoders.pkl`.
- `requirements.txt` — Python package dependencies.
- `Procfile` — Process definition for deployment.
- `data/` — Datasets (`Expanded_Destinations.csv`, `Final_Updated_Expanded_UserHistory.csv`, `final_df.csv`).
- `models/` — Serialized model (`model.pkl`) and label encoders (`label_encoders.pkl`).
- `templates/` — HTML templates (`index.html`, `recommendation.html`).

## Setup & Running Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
