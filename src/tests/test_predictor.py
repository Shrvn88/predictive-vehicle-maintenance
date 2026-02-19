from src.data.cmapss_loader import load_cmapss
from src.models.predict import RULPredictor

# Load CMAPSS data
train_df, _ = load_cmapss()

# Take one engine with enough cycles for rolling windows
unit_id = train_df["unit"].iloc[0]
sample = train_df[train_df["unit"] == unit_id].head(200)

print("Input sample shape:", sample.shape)
print("Cycles range:", sample["cycle"].min(), "→", sample["cycle"].max())

# Initialize predictor
predictor = RULPredictor()

# Run inference
out = predictor.predict_from_dataframe(sample)

print("\nPrediction output head:")
print(out.head())

print("\nPrediction output tail:")
print(out.tail())

# Basic sanity checks
assert "predicted_RUL" in out.columns
assert out["predicted_RUL"].between(0, 125).all()

print("\nMin predicted RUL:", out["predicted_RUL"].min())
print("Max predicted RUL:", out["predicted_RUL"].max())

# Optional: check monotonic trend roughly decreases
print("\nFirst vs last prediction:")
print(out.iloc[0]["predicted_RUL"], "→", out.iloc[-1]["predicted_RUL"])
