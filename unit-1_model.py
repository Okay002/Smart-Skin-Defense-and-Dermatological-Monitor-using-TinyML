import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
df = pd.read_csv("updated_uv_exposure_dataset.csv")

# Display dataset information
print("Dataset Information:\n")
df.info()
print("\nFirst 5 Rows:\n", df.head())

# Encode target variable
label_encoder = LabelEncoder()
df["UV_RISK_LEVEL"] = label_encoder.fit_transform(df["UV_RISK_LEVEL"])
num_classes = len(label_encoder.classes_)

# Standardize numerical features
scaler = StandardScaler()
numerical_features = ["Temperature (°C)", "Humidity (%)", "Light Intensity (lux)", "UV Index"]
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# Visualize class distribution
plt.figure(figsize=(8,5))
sns.countplot(x=df["UV_RISK_LEVEL"], palette="viridis")
plt.title("UV Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.show()

# Split dataset into features and target
X = df.drop(columns=["UV_RISK_LEVEL"])
y = keras.utils.to_categorical(df["UV_RISK_LEVEL"], num_classes=num_classes)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df["UV_RISK_LEVEL"])

# Define Keras model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    BatchNormalization(),
    Dropout(0.4),
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(num_classes, activation='softmax')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=75, batch_size=16, validation_data=(X_test, y_test))

# Evaluate model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)
report = classification_report(y_test_labels, y_pred, target_names=label_encoder.classes_)
cm = confusion_matrix(y_test_labels, y_pred)

# Print results
print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report:\n", report)

# Plot accuracy and loss
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

plot_training_history(history)

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Convert to TensorFlow Lite model
tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = tflite_converter.convert()
with open("model_uv.tflite", "wb") as f:
    f.write(tflite_model)

# Convert to C header file
with open("model_uv.h", "w") as f:
    f.write("const unsigned char model_uv[] = {\n")
    f.write(",".join(map(str, tflite_model)) + "\n};\n")
    f.write("const unsigned int model_uv_len = " + str(len(tflite_model)) + ";\n")