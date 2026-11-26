import os
import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_df.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# 1. Load dataset
df = pd.read_csv(DATA_PATH)

features = [
    "Name_x",
    "State",
    "Type",
    "BestTimeToVisit",
    "Preferences",
    "Gender",
    "NumberOfAdults",
    "NumberOfChildren",
]
target = "Popularity"

df = df.dropna(subset=features + [target]).copy()

# 2. Encode text columns into numbers
categorical_cols = [
    "Name_x",
    "State",
    "Type",
    "BestTimeToVisit",
    "Preferences",
    "Gender",
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Train new model with your local scikit-learn 1.7.0
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Overwrite old .pkl files with new compatible ones
os.makedirs(MODEL_DIR, exist_ok=True)
with open(os.path.join(MODEL_DIR, "model.pkl"), "wb") as f:
    pickle.dump(model, f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "wb") as f:
    pickle.dump(label_encoders, f)

print("SUCCESS! Fresh model.pkl and label_encoders.pkl saved in models/")