#include <Arduino.h>
#include <DHT.h>
#include <TensorFlowLite_ESP32.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model.h"  // Include the model header file

// Define sensor pins
#define DHT_PIN 16  // Change this if needed
#define PHOTO_PIN 34
#define BUZZER_PIN 5

// Setup DHT sensor
DHT dht(DHT_PIN, DHT11);

// Increase tensor arena size
constexpr int kTensorArenaSize = 40 * 1024; // Increased to 40 KB
uint8_t tensor_arena[kTensorArenaSize];

// Error reporter and model interpreter
static tflite::MicroErrorReporter error_reporter;
const tflite::Model* model = tflite::GetModel(model_tflite);
tflite::AllOpsResolver resolver;
static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize, &error_reporter);
tflite::MicroInterpreter* tflite_interpreter = &interpreter;

void resetDHT() {
    Serial.println("🔄 Resetting DHT sensor...");
    pinMode(DHT_PIN, OUTPUT);
    digitalWrite(DHT_PIN, LOW);
    delay(1000);
    digitalWrite(DHT_PIN, HIGH);
    delay(1000);
    dht.begin();
}

void setup() {
    Serial.begin(115200);
    pinMode(BUZZER_PIN, OUTPUT);
    
    dht.begin();
    delay(2000); // Allow sensor to stabilize

    Serial.print("Free Heap Before Model Load: ");
    Serial.println(ESP.getFreeHeap());

    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("❌ Model schema version mismatch!");
        while (1);
    }

    if (tflite_interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("❌ Failed to allocate tensors!");
        while (1);
    }

    Serial.println("✅ Tensors allocated successfully.");
    Serial.print("TFLite Model Version: ");
    Serial.println(model->version());
}

void loop() {
    Serial.print("Free Heap Before Inference: ");
    Serial.println(ESP.getFreeHeap());

    // Read sensor data
    float humidity = dht.readHumidity();
    float temperature = dht.readTemperature();
    int uvIndexRaw = analogRead(PHOTO_PIN);
    float uvIndex = map(uvIndexRaw, 0, 4095, 0, 10);

    // Check if DHT sensor reading is valid
    if (isnan(humidity) || isnan(temperature)) {
        Serial.println("❌ Failed to read from DHT sensor! Retrying...");
        resetDHT();  // Reset the sensor and retry
        delay(2000);
        return;
    }

    Serial.print("🌡️ Temp: "); Serial.print(temperature); Serial.print("°C, ");
    Serial.print("💧 Humidity: "); Serial.print(humidity); Serial.print("%, ");
    Serial.print("☀️ UV Index: "); Serial.println(uvIndex);

    // Get input tensor
    TfLiteTensor* input_tensor = tflite_interpreter->input(0);
    if (!input_tensor) {
        Serial.println("❌ Failed to get input tensor!");
        return;
    }

    if (input_tensor->type != kTfLiteFloat32) {
        Serial.println("❌ Model expects different input type!");
        return;
    }

    float* input_data = input_tensor->data.f;
    input_data[0] = temperature;
    input_data[1] = humidity;
    input_data[2] = uvIndex;

    // Perform inference
    if (tflite_interpreter->Invoke() != kTfLiteOk) {
        Serial.println("❌ Model inference failed!");
        return;
    }
    
    Serial.println("✅ Inference successful.");

    // Get output tensor
    TfLiteTensor* output_tensor = tflite_interpreter->output(0);
    if (!output_tensor) {
        Serial.println("❌ Failed to get output tensor!");
        return;
    }

    // Check for valid output data type
    if (output_tensor->type != kTfLiteFloat32) {
        Serial.println("❌ Unsupported output data type!");
        return;
    }

    float* output_data = output_tensor->data.f;
    int predicted_class = std::distance(output_data, std::max_element(output_data, output_data + 4));

    Serial.print("🔴 Predicted UV Risk Level: ");
    int buzz_count = 0;

    switch (predicted_class) {
        case 0: 
            Serial.println("Low");
            buzz_count = 0;
            break;
        case 1: 
            Serial.println("Moderate");
            buzz_count = 1;
            break;
        case 2: 
            Serial.println("High");
            buzz_count = 2;
            break;
        case 3: 
            Serial.println("Very High");
            buzz_count = 3;
            break;
    }

    // Activate buzzer based on predicted class
    for (int i = 0; i < buzz_count; i++) {
        digitalWrite(BUZZER_PIN, HIGH);
        delay(500);
        digitalWrite(BUZZER_PIN, LOW);
        delay(500);
    }

    delay(2000);
}

