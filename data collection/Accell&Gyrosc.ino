/*
  Arduino LSM9DS1 - Accelerometer + Gyroscope data collection

  This example reads the acceleration and gyroscope values from the LSM9DS1
  sensor and continuously prints them to the Serial Monitor
  or Serial Plotter.

  The circuit:
  - Arduino Nano 33 BLE Sense

  created 10 Jul 2019
  by Riccardo Rizzo and modified by "I Cioccolatosi" team

  This example code is in the public domain.
*/
#include <Arduino_LSM9DS1.h>


void setup() {
  Serial.begin(9600);
  while (!Serial);
  Serial.println("Started");

  if (!IMU.begin()) {
    Serial.println("Failed to initialize IMU!");
    while (1);
  }

  /*Serial.print("Accelerometer sample rate = ");
  Serial.print("Gyroscope sample rate = ");

  Serial.print(IMU.accelerationSampleRate());
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  Serial.println();

  Serial.println("Acceleration in m/s^2");
  Serial.println("Gyroscope in degrees/second");
  */

  Serial.println("xAcc\tyAcc\tzAcc\txGyro\tyGyro\tzGyro");
}

void loop() {
  float xAcc, yAcc, zAcc;
  float xGyro, yGyro, zGyro;

  if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
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
  }
}
