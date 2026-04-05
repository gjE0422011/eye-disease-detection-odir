from model import build_model
from preprocessing import load_dataset
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Load data
X, y = load_dataset('../data/')
y = to_categorical(y, num_classes=8)

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Build and train
model = build_model(num_classes=8)
model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

# Save model
model.save('../models/eye_disease_model.h5')
print("✅ Model saved successfully!")
