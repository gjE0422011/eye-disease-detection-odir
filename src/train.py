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

import matplotlib.pyplot as plt

def save_training_graphs(history):
    # Accuracy Graph
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Graph
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('../results/graphs/training_graphs.png')
    plt.close()
    print("✅ Training graphs saved!")

# Update your model.fit to capture history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_test, y_test)
)

save_training_graphs(history)
