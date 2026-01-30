#include <iostream>
#include <vector>
#include <math.h>
#include <array>
#include <string>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <random>
#include <chrono>
#include <Eigen/Dense>

using namespace std;
namespace e = Eigen;

#define inputSize 784 //maybe some geometric progression going on
#define layer2size 183
#define layer3size 43
#define outputSize 10

// uint32_t reverse_bytes(uint32_t bytes) {
//     return (bytes >> 24) | ((bytes >> 8) & 0x0000FF00) | ((bytes << 8) & 0x00FF0000) | (bytes << 24);
// }

// e::VectorXd imageToVector(const string& imagePath, int size){

//     ifstream file(imagePath);
//     e::VectorXd normValues;
//     string line;
//     vector<double> values;

//     if (!file.is_open()) {
//         cerr << "Error opening file: " << imagePath << endl;
//         return normValues;
//     }

//     while (getline(file, line)){

//     }

    // uint32_t magic_number = 0;
    // uint32_t num_images = 0;
    // uint32_t num_rows = 0;
    // uint32_t num_cols = 0;

    // file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    // magic_number = reverse_bytes(magic_number);

    // file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    // num_images = reverse_bytes(num_images);

    // file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    // num_rows = reverse_bytes(num_rows);

    // file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    // num_cols = reverse_bytes(num_cols);

    // for (int i = 0; i < size; ++i) {
    //     unsigned char pixel_value;
    //     file.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
        
    //     // normalize pixel values from [0, 255] to [0.0, 1.0]
    //     normValues << static_cast<double>(pixel_value) / 255.0;
    // }

    // file.close();
    // return normValues;

// }

double sigmoid(double x){ return 1.0 / (1.0 + exp(-x)); }

double sigmoidPrime(double x){ return sigmoid(x) * (1 - sigmoid(x)); }