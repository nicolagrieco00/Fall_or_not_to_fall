class StandardScaler {
private:
    float mean[6]; // Assuming you have 6 features
    float stdDev[6]; // Assuming you have 6 features

public:
    void fit(float* data, int numSamples, int numFeatures);
    void transform(float* data, int numSamples, int numFeatures);
};