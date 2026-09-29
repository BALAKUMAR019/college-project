import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# Smart Plant Monitoring - Deep Learning Model
# Python 3.13 compatible version
# No scikit-learn / scipy required
# ============================================================

np.random.seed(42)
tf.random.set_seed(42)

print("Creating synthetic plant sensor dataset...")


# ------------------------------------------------------------
# 1. Generate sensor data
# ------------------------------------------------------------

N = 3000

soil_moisture = np.random.uniform(0, 100, N)
temperature = np.random.uniform(15, 40, N)
humidity = np.random.uniform(25, 90, N)

X = np.column_stack([
    soil_moisture,
    temperature,
    humidity
]).astype(np.float32)


# ------------------------------------------------------------
# 2. Create plant condition labels
#
# 0 = Healthy
# 1 = Needs Water
# 2 = Dry / Risk
# ------------------------------------------------------------

labels = np.zeros(N, dtype=np.int32)

dry_condition = soil_moisture < 25

water_condition = (
    ((soil_moisture >= 25) & (soil_moisture < 40))
    |
    ((temperature > 36) & (humidity < 40))
)

labels[water_condition] = 1
labels[dry_condition] = 2


# ------------------------------------------------------------
# 3. Normalize data manually
# ------------------------------------------------------------

mean = X.mean(axis=0)
std = X.std(axis=0)

X_scaled = (X - mean) / std


# Save scaler information
scaler_data = {
    "mean": mean,
    "std": std
}

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler_data, f)

print("Scaler saved: scaler.pkl")


# ------------------------------------------------------------
# 4. Train / validation split
# ------------------------------------------------------------

split = int(N * 0.8)

indices = np.random.permutation(N)

train_idx = indices[:split]
test_idx = indices[split:]

X_train = X_scaled[train_idx]
X_test = X_scaled[test_idx]

y_train = labels[train_idx]
y_test = labels[test_idx]


# ------------------------------------------------------------
# 5. Build Deep Learning model
# ------------------------------------------------------------

model = Sequential([
    Dense(32, activation="relu", input_shape=(3,)),
    Dropout(0.20),

    Dense(16, activation="relu"),
    Dropout(0.15),

    Dense(8, activation="relu"),

    Dense(3, activation="softmax")
])


model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


model.summary()


# ------------------------------------------------------------
# 6. Train
# ------------------------------------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


print("\nTraining model...\n")

history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=30,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)


# ------------------------------------------------------------
# 7. Evaluate
# ------------------------------------------------------------

loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n==============================")
print("MODEL TRAINING COMPLETED")
print("==============================")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print(f"Test Loss: {loss:.4f}")


# ------------------------------------------------------------
# 8. Save model
# ------------------------------------------------------------

model.save("plant_model.keras")

print("\nModel saved as:")
print("plant_model.keras")


# ------------------------------------------------------------
# 9. Test sample predictions
# ------------------------------------------------------------

sample_data = np.array([
    [62, 24, 68],   # Healthy
    [32, 25, 60],   # Needs Water
    [15, 30, 45]    # Dry / Risk
], dtype=np.float32)

sample_scaled = (sample_data - mean) / std

predictions = model.predict(sample_scaled, verbose=0)

class_names = [
    "Healthy",
    "Needs Water",
    "Dry / Risk"
]

print("\nSample Predictions:")

for i, prediction in enumerate(predictions):
    predicted_class = np.argmax(prediction)

    print(
        f"Soil={sample_data[i][0]:.0f}% | "
        f"Temp={sample_data[i][1]:.0f}C | "
        f"Humidity={sample_data[i][2]:.0f}% "
        f"--> {class_names[predicted_class]}"
    )

print("\nAll files generated successfully!")