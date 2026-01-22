#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <array>
#include <string>
#include <unistd.h>
#include <fstream>
#include <cstdint>
#include <random>
#include <chrono>

#include "helpers.h"

using namespace std;

#define inputSize 784 //maybe some geometric progression going on
#define layer2size 183
#define layer3size 43
#define outputSize 10

// create a network class so it makes life easier
class Network {

    public:
        vector<double> input;
        vector<double> layer2;
        vector<double> layer3;
        vector<double> output;
        vector<double> b2;
        vector<double> b3;
        vector<double> b4;
        array<array<float, inputSize>, layer2size> w12; // there are (layer2size) rows of (inputSize) weights so you can matrix multiply
        array<array<float, layer2size>, layer3size> w23; // same for these
        array<array<float, layer3size>, outputSize> w34;

        Network(){
            fill(layer2.begin(), layer2.end(), 0);
            fill(layer3.begin(), layer3.end(), 0);
            fill(b2.begin(), b2.end(), 0);
            fill(b3.begin(), b3.end(), 0);
            fill(b4.begin(), b4.end(), 0);
        }

        void reset(){
            fill(input.begin(), input.end(), 0);
            fill(layer2.begin(), layer2.end(), 0);
            fill(layer3.begin(), layer3.end(), 0);
            fill(output.begin(), output.end(), 0);
        }

        void randomize(){
            unsigned seed = chrono::system_clock::now().time_since_epoch().count();
            mt19937 gen(seed);
            uniform_int_distribution<int> distrib(-0.5, 0.5);

            for (auto& e1 : w12){
                for (auto& e2 : e1){
                    e2 = distrib(gen);
                }
            }
            for (auto& e1 : w23){
                for (auto& e2 : e1){
                    e2 = distrib(gen);
                }
            }
            for (auto& e1 : w34){
                for (auto& e2 : e1){
                    e2 = distrib(gen);
                }
            }
        }

        void fillWeights(double x){
            for (auto& e1 : w12){
                for (auto& e2 : e1){
                    e2 = x;
                } 
            }
            for (auto& e1 : w23){
                for (auto& e2 : e1){
                    e2 = x;
                } 
            }
            for (auto& e1 : w34){
                for (auto& e2 : e1){
                    e2 = x;
                } 
            }
        }

        vector<double> identify(const string& imagePath){
            reset();
            input = imageToVector(imagePath, inputSize);

            // all of the matrix multiplication
            for (int i = 0; i < w12.size(); ++i){
                for (int j = 0; j < input.size(); ++j){
                    layer2[i] += (w12[i][j] * input[j]); 
                }
                layer2[i] += b2[i];
            }
            for (int i = 0; i < layer3size; ++i){
                for (int j = 0; j < layer2size; ++j){
                    layer3[i] += (w23[i][j] * layer2[j]); 
                }
                layer3[i] += b3[i];
            }

            for (int i = 0; i < outputSize; ++i){
                for (int j = 0; j < layer3size; ++j){
                    output[i] += (w34[i][j] * layer3[j]); 
                }
                output[i] = sigmoid(output[i] + b4[i]);
            }

            return output;
        }

        double cost(vector<double> identification, vector<double> target){
            double cost = 0;
            for (int i = 0; i < identification.size(); ++i){
                cost += pow((identification[i] - target[i]), 2);
            }
            return cost;
        }

        void backpropogate(double cost, double stepsize){

        }
};

int main(int argc, char *argv[]){
    /*
    // OLD - NOT USED ANYMORE
    // random stuff
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<int> distrib(-0.5, 0.5);

    // intialize weights and biases
    vector<double> input;
    vector<double> biases2(layer2size, 0.0);
    vector<double> biases3(layer3size, 0.0);
    vector<double> biases4(outputSize, 0.0);
    array<array<float, inputSize>, layer2size> weights12; // there are (layer2size) rows of (inputSize) weights so you can matrix multiply
    array<array<float, layer2size>, layer3size> weights23; // same for these
    array<array<float, layer3size>, outputSize> weights34;

    // initialize with random values
    for (auto& e1 : weights12){
        for (auto& e2 : e1){
            e2 = distrib(gen);
        }
    }
    for (auto& e1 : weights23){
        for (auto& e2 : e1){
            e2 = distrib(gen);
        }
    }
    for (auto& e1 : weights34){
        for (auto& e2 : e1){
            e2 = distrib(gen);
        }
    }
    */
    Network network;

    ifstream map("map.txt");

    if (!map.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    string line;
    double cost;
    string path;
    vector<double> target(outputSize, 0.0);
    vector<double> identification;

    double rate = 0.02;
    Network change;
    change.fillWeights(0.0);
    double z = 0.0;

    while (getline(map, line)){
        path = line.substr(0, line.find(','));
        fill(target.begin(), target.end(), 0);
        target[stoi(line.substr(line.find(',')))] = 1.0;

        identification = network.identify(path);

        for (int j = 0; j < size(change.w34); ++j){
            for (int k = 0; k < size(change.w34[j]); ++k){
                z += (network.w34[j][k] * network.layer3[k]) + network.b4[k];
            }
            for (int k = 0; k < size(change.w34[j]); ++k){
                change.w34[j][k] += -1 * rate * network.layer3[k] * sigmoidPrime(z) * 2 * (identification[j] - target[j]);
                change.layer2[k] += -1 * rate * network.w34[j][k] * sigmoidPrime(z) * 2 * (identification[j] - target[j]);
            }
            change.b4[j] += -1 * rate * sigmoidPrime(z) * 2 * (identification[j] - target[j]);
            z = 0.0;
        }
    }
    
    return 0;
}