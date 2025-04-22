// #include <DHT.h>
// #include <DHT_U.h>

// #include <DHT.h>
// #include <TensorFlowLite.h>  // TensorFlow Lite library
// #include "uv_model.h"  // Include the header file for the .tflite model


// // Define sensor pins
// #define DHT_PIN 16
// #define PHOTO_PIN 34
// #define BUZZER_PIN 5

// // Setup DHT sensor
// DHT dht(DHT_PIN, DHT11);

// // Define the model input and output tensor size
// #define NUM_INPUTS 2
// #define NUM_CLASSES 5  // UV risk levels: Low, Moderate, High, Very High
// #define MODEL_INPUT_SIZE (NUM_INPUTS * sizeof(float))
// #define MODEL_OUTPUT_SIZE (NUM_CLASSES * sizeof(float))

// // Create a TensorFlow Lite interpreter
// tflite::MicroInterpreter* interpreter;
// tflite::MicroOpResolver<5> resolver;
// tflite::MicroErrorReporter error_reporter;

// // Allocate memory for input and output tensors
// float input_data[NUM_INPUTS];  // Sensor inputs: UV index and temperature
// float output_data[NUM_CLASSES]; // Prediction output: UV risk level

// // Setup model and tensors
// void setupModel() {
//   static tflite::MicroInterpreter* interpreter = nullptr;
//   static tflite::MicroAllocator* allocator = nullptr;

//   // Load the TensorFlow Lite model into memory
//   interpreter = new tflite::MicroInterpreter(model, resolver, allocator, &error_reporter);
//   interpreter->AllocateTensors();
// }

// void setup() {
//   Serial.begin(115200);
//   pinMode(BUZZER_PIN, OUTPUT);
//   dht.begin();
  
//   // Initialize TensorFlow Lite model
//   setupModel();
// }

// void loop() {
//   // Read sensor data
//   float humidity = dht.readHumidity();
//   float temperature = dht.readTemperature();
//   int uvIndex = analogRead(PHOTO_PIN);  // Simulate UV index from photoresistor (0-1023)

//   // Normalize UV index (map to 0-100 scale)
//   float normalizedUV = map(uvIndex, 0, 1023, 0, 100);
//   input_data[0] = normalizedUV;        // UV Index as input
//   input_data[1] = temperature;        // Temperature as input

//   // Copy inputs to model's input tensor
//   memcpy(interpreter->input(0)->data.f, input_data, sizeof(input_data));

//   // Perform inference
//   interpreter->Invoke();

//   // Get the output of the model (predicted risk level)
//   memcpy(output_data, interpreter->output(0)->data.f, sizeof(output_data));

//   // Find the class with the highest probability
//   int predictedClass = 0;
//   float maxProb = output_data[0];
//   for (int i = 1; i < NUM_CLASSES; i++) {
//     if (output_data[i] > maxProb) {
//       maxProb = output_data[i];
//       predictedClass = i;
//     }
//   }

//   // Print the predicted UV risk level
//   Serial.print("Predicted UV Risk Level: ");
//   switch (predictedClass) {
//     case 0: Serial.println("Low"); break;
//     case 1: Serial.println("Moderate"); break;
//     case 2: Serial.println("High"); break;
//     case 3: Serial.println("Very High"); break;
//   }

//   // Trigger buzzer for high risk levels
//   if (predictedClass == 2 || predictedClass == 3) {
//     digitalWrite(BUZZER_PIN, HIGH);
//     delay(1000);  // Buzzer on for 1 second
//     digitalWrite(BUZZER_PIN, LOW);
//   }

//   delay(2000);  // Delay before next reading
// }
