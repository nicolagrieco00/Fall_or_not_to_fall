#include "StandardScaler.h"
#include <cmath>

void StandardScaler::fit(float* data, int numSamples, int numFeatures) {
    for (int i = 0; i < numFeatures; i++) {
        float sum = 0.0;
        for (int j = 0; j < numSamples; j++) {
            sum += data[j * numFeatures + i];
        }
        mean[i] = sum / numSamples;
    }

    for (int i = 0; i < numFeatures; i++) {
        float sumSquares = 0.0;
        for (int j = 0; j < numSamples; j++) {
            float diff = data[j * numFeatures + i] - mean[i];
            sumSquares += diff * diff;
        }
        stdDev[i] = sqrt(sumSquares / numSamples);
    }
}

void StandardScaler::transform(float* data, int numSamples, int numFeatures) {
    for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
            data[i * numFeatures + j] = (data[i * numFeatures + j] - mean[j]) / stdDev[j];
        }
    }
}