import joblib
import os

# paths
MODEL_PATH = "model1.pkl"
COMPRESSED_PATH = "best_model_compressed.pkl"

# load original model
print("Loading model...")
model = joblib.load(MODEL_PATH)

# save compressed model
print("Saving compressed model...")
joblib.dump(model, COMPRESSED_PATH, compress=3)

# check size difference
original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
compressed_size = os.path.getsize(COMPRESSED_PATH) / (1024 * 1024)

print(f"Original size: {original_size:.2f} MB")
print(f"Compressed size: {compressed_size:.2f} MB")
