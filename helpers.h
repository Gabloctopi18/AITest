#ifndef HELPERS_H
#define HELPERS_H

#include <vector>
#include <array>
#include <string>
#include <Eigen/Dense>

// #define inputSize 784 //maybe some geometric progression going on
// #define layer2size 183
// #define layer3size 43
// #define outputSize 10

using namespace std;
namespace e = Eigen;

e::VectorXd imageToVector(const string& imagePath, int size);
double sigmoid(double x);
double sigmoidPrime(double x);

#endif HELPERS_H