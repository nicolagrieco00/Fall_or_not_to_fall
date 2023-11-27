/*

*/

#include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"
#include "StandardScaler.h"

const int numTrainSamples = 400;
const int numFeatures = 6;

// global variables used for TensorFlow Lite (Micro)
tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops
tflite::AllOpsResolver tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* STATUS[] = {
  "fall",
  "rfall",
  "lfall",
  "light",
  "sit",
  "walk",
  "step"
};

#define NUM_STATUS (sizeof(STATUS) / sizeof(STATUS[0]))
//StandardScaler scaler;

void setup() {
  // init the serial
  Serial.begin(9600);
  while (!Serial);

  // initialize the IMU
  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  // print out the samples rates of the IMUs
  Serial.print("Accelerometer sample rate = ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("Gyroscope sample rate = ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  Serial.println();

  // get the TFL representation of the model byte array
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);

  // Allocate memory for the model's input and output tensors
  tflInterpreter->AllocateTensors();

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

}

void loop() {

  float accData[400][3]; // Assuming 400 time steps and 3 acceleration values (x, y, z)
  float gyroData[400][3]; // Assuming 400 time steps and 3 gyroscope values (x, y, z)
  // Initialize an index for data insertion
  int insertIndex = 0;

  // Check if new acceleration AND gyroscope data is available
  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
    // Read the acceleration and gyroscope data
    for (int i = 0; i < 2000; i++) {
      float xAcc, yAcc, zAcc;
      float xGyro, yGyro, zGyro;

      // Read acceleration and gyroscope data
      IMU.readAcceleration(xAcc, yAcc, zAcc);
      IMU.readGyroscope(xGyro, yGyro, zGyro);
      Serial.print(xAcc*9.81);
      Serial.print('\t');
      Serial.print(yAcc*9.81);
      Serial.print('\t');
      Serial.print(zAcc*9.81);
      Serial.print('\t');
      Serial.print(xGyro);
      Serial.print('\t');
      Serial.print(yGyro);
      Serial.print('\t');
      Serial.print(zGyro);
      Serial.println();

      // Check if (i-1) is divisible by 5 ((i-1) % 5 == 0)
      if (i % 5 == 0) {
        accData[insertIndex][0] = xAcc*9.81;
        accData[insertIndex][1] = yAcc*9.81;
        accData[insertIndex][2] = zAcc*9.81;
        gyroData[insertIndex][0] = xGyro*9.81;
        gyroData[insertIndex][1] = yGyro*9.81;
        gyroData[insertIndex][2] = zGyro*9.81;

        insertIndex++;
      }

      // Stop if we've filled 400 samples
      if (insertIndex >= 400) {
        break;
      }
    }

    //// Fit the scaler on all the data
    //scaler.fit(&accData[0][0], 400, 3);
    //scaler.fit(&gyroData[0][0], 400, 3);
//
    //// Transform the data
    //scaler.transform(&accData[0][0], 400, 3);
    //scaler.transform(&gyroData[0][0], 400, 3);

    // Populate the input tensor with the scaled data
    for (int i = 0; i < 400; i++) {
      for (int j = 0; j < 3; j++) {
        tflInputTensor->data.f[i * 6 + j] = accData[i][j];
        tflInputTensor->data.f[i * 6 + j + 3] = gyroData[i][j];
      }
    }

    // Run inference
    TfLiteStatus invokeStatus = tflInterpreter->Invoke();
    if (invokeStatus != kTfLiteOk) {
      Serial.println("Invoke failed!");
      while (1);
      return;
    }

    // Loop through the output tensor values from the model
    for (int i = 0; i < NUM_STATUS; i++) {
      Serial.print(STATUS[i]);
      Serial.print(": ");
      Serial.println(tflOutputTensor->data.f[i], 6);
    }
    Serial.println();
  }
}
