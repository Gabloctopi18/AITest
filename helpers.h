#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <array>
#include <string>

#define inputSize 784 //maybe some geometric progression going on
#define layer2size 183
#define layer3size 43
#define outputSize 10

using namespace std;

vector<double> imageToVector(const string& imagePath, int size);
double sigmoid(double x);
vector<double> identify(const string& imagePath, 
    array<array<float, inputSize>, layer2size> weights12, 
    array<array<float, layer2size>, layer3size> weights23, 
    array<array<float, layer3size>, outputSize> weights34, 
    vector<double> biases2, 
    vector<double> biases3, 
    vector<double> biases4);

#endif HELPERS_H