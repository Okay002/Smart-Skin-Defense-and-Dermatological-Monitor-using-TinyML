# from pyspark.sql import SparkSession
# from pyspark.sql.functions import col
# from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler as SparkStandardScaler

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
# from tensorflow.keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import classification_report, confusion_matrix
# import os

# # === 1. Initialize Spark with config ===
# spark = SparkSession.builder \
#     .appName("UV_Exposure_Prediction") \
#     .config("spark.sql.repl.eagerEval.enabled", True) \
#     .config("spark.driver.extraJavaOptions", "-Duser.name=your-username") \
#     .getOrCreate()


# # Add the Spark Java options (ensure the username is correct)
# #spark.conf.set("spark.driver.extraJavaOptions", "-Duser.name=your-username")  # Update with actual username if needed

# # === 2. Load CSV (Ensure file exists in path) ===
# csv_path = "updated_uv_exposure_dataset.csv"
# if not os.path.exists(csv_path):
#     raise FileNotFoundError(f"CSV file not found: {csv_path}")

# df_spark = spark.read.csv(csv_path, header=True, inferSchema=True)

# # === 3. Label encode target ===
# indexer = StringIndexer(inputCol="UV_RISK_LEVEL", outputCol="UV_RISK_LEVEL_INDEX")
# df_spark = indexer.fit(df_spark).transform(df_spark)

# # === 4. Assemble and Scale numerical features ===
# numerical_features = ["Temperature (°C)", "Humidity (%)", "Light Intensity (lux)", "UV Index"]
# assembler = VectorAssembler(inputCols=numerical_features, outputCol="features_vec")
# df_spark = assembler.transform(df_spark)

# scaler = SparkStandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
# scaler_model = scaler.fit(df_spark)
# df_spark = scaler_model.transform(df_spark)

# # === 5. Convert Spark DataFrame to Pandas ===
# pandas_df = df_spark.select("scaled_features", "UV_RISK_LEVEL_INDEX").toPandas()
# X_scaled = np.array(pandas_df["scaled_features"].tolist())
# y_labels = pandas_df["UV_RISK_LEVEL_INDEX"].astype(int).values

# # === 6. Encode target for classification ===
# label_encoder = LabelEncoder()
# y = label_encoder.fit_transform(y_labels)
# y_categorical = keras.utils.to_categorical(y)
# num_classes = y_categorical.shape[1]

# # === 7. Visualize Class Distribution ===
# plt.figure(figsize=(8, 5))
# sns.countplot(x=y, palette="viridis")
# plt.title("UV Risk Level Distribution")
# plt.xlabel("Risk Level")
# plt.ylabel("Count")
# plt.show()

# # === 8. Train/Test Split ===
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y)

# # === 9. Define Model ===
# model = Sequential([
#     Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
#     BatchNormalization(),
#     Dropout(0.4),
#     Dense(64, activation='relu'),
#     BatchNormalization(),
#     Dropout(0.3),
#     Dense(32, activation='relu'),
#     Dropout(0.3),
#     Dense(num_classes, activation='softmax')
# ])
# model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# # === 10. Train ===
# history = model.fit(X_train, y_train, epochs=75, batch_size=16, validation_data=(X_test, y_test))

# # === 11. Evaluate ===
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# y_pred = np.argmax(model.predict(X_test), axis=1)
# y_test_labels = np.argmax(y_test, axis=1)
# report = classification_report(y_test_labels, y_pred, target_names=label_encoder.classes_)
# cm = confusion_matrix(y_test_labels, y_pred)

# print(f"Test Accuracy: {test_accuracy:.4f}")
# print("Classification Report:\n", report)

# # === 12. Plot Accuracy and Loss ===
# def plot_training_history(history):
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Training Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
#     plt.title('Model Accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Training Loss')
#     plt.plot(history.history['val_loss'], label='Validation Loss')
#     plt.title('Model Loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     plt.show()

# plot_training_history(history)

# # === 13. Confusion Matrix ===
# plt.figure(figsize=(6, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title("Confusion Matrix")
# plt.xlabel("Predicted")
# plt.ylabel("Actual")
# plt.show()

# # === 14. Convert to TFLite ===
# # tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
# # tflite_model = tflite_converter.convert()
# # with open("model_uv.tflite", "wb") as f:
# #     f.write(tflite_model)

# # # === 15. Convert to C header ===
# # with open("model_uv.h", "w") as f:
# #     f.write("const unsigned char model_uv[] = {\n")
# #     f.write(",".join(map(str, tflite_model)) + "\n};\n")
# #     f.write("const unsigned int model_uv_len = " + str(len(tflite_model)) + ";\n")
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler as SparkStandardScaler

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
from sklearn.metrics import classification_report, confusion_matrix
import os

# === 1. Initialize Spark with config ===
spark = SparkSession.builder \
    .appName("UV_Exposure_Prediction") \
    .config("spark.sql.repl.eagerEval.enabled", True) \
    .config("spark.driver.extraJavaOptions", "-Duser.name=your-username") \
    .getOrCreate()

# === 2. Load CSV (Ensure file exists in path) ===
csv_path = "updated_uv_exposure_dataset.csv"
if not os.path.exists(csv_path):
    raise FileNotFoundError(f"CSV file not found: {csv_path}")

df_spark = spark.read.csv(csv_path, header=True, inferSchema=True)

# === 3. Label encode target ===
indexer = StringIndexer(inputCol="UV_RISK_LEVEL", outputCol="UV_RISK_LEVEL_INDEX")
indexer_model = indexer.fit(df_spark)
df_spark = indexer_model.transform(df_spark)
label_names = indexer_model.labels  # ['LOW', 'MODERATE', 'HIGH', 'VERY HIGH']

# === 4. Assemble and Scale numerical features ===
numerical_features = ["Temperature (°C)", "Humidity (%)", "Light Intensity (lux)", "UV Index"]
assembler = VectorAssembler(inputCols=numerical_features, outputCol="features_vec")
df_spark = assembler.transform(df_spark)

scaler = SparkStandardScaler(inputCol="features_vec", outputCol="scaled_features", withMean=True, withStd=True)
scaler_model = scaler.fit(df_spark)
df_spark = scaler_model.transform(df_spark)

# === 5. Convert Spark DataFrame to Pandas ===
pandas_df = df_spark.select("scaled_features", "UV_RISK_LEVEL_INDEX").toPandas()
X_scaled = np.array(pandas_df["scaled_features"].tolist())
y_labels = pandas_df["UV_RISK_LEVEL_INDEX"].astype(int).values

# === 6. Convert to categorical ===
y_categorical = keras.utils.to_categorical(y_labels)
num_classes = y_categorical.shape[1]

# === 7. Visualize Class Distribution ===
plt.figure(figsize=(8, 5))
sns.countplot(x=y_labels, palette="viridis")
plt.title("UV Risk Level Distribution")
plt.xlabel("Risk Level")
plt.ylabel("Count")
plt.xticks(ticks=np.arange(len(label_names)), labels=label_names)
plt.show()

# === 8. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_categorical, test_size=0.2, random_state=42, stratify=y_labels)

# === 9. Define Model ===
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
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# === 10. Train ===
history = model.fit(X_train, y_train, epochs=75, batch_size=16, validation_data=(X_test, y_test))

# === 11. Evaluate ===
test_loss, test_accuracy = model.evaluate(X_test, y_test)
y_pred = np.argmax(model.predict(X_test), axis=1)
y_test_labels = np.argmax(y_test, axis=1)

report = classification_report(y_test_labels, y_pred, target_names=label_names)
cm = confusion_matrix(y_test_labels, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print("Classification Report:\n", report)

# === 12. Plot Accuracy and Loss ===
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

# === 13. Confusion Matrix ===
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# === 14. Convert to TFLite ===
# tflite_converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = tflite_converter.convert()
# with open("model_uv.tflite", "wb") as f:
#     f.write(tflite_model)

# === 15. Convert to C header ===
# with open("model_uv.h", "w") as f:
#     f.write("const unsigned char model_uv[] = {\n")
#     f.write(",".join(map(str, tflite_model)) + "\n};\n")
#     f.write("const unsigned int model_uv_len = " + str(len(tflite_model)) + ";\n")