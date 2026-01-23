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
        vector<double> l2;
        vector<double> l3;
        vector<double> output;
        vector<double> b2;
        vector<double> b3;
        vector<double> b4;
        array<array<float, inputSize>, layer2size> w12; // there are (layer2size) rows of (inputSize) weights so you can matrix multiply
        array<array<float, layer2size>, layer3size> w23; // same for these
        array<array<float, layer3size>, outputSize> w34;

        Network():
            l2(layer2size), 
            l3(layer3size), 
            b2(layer2size), 
            b3(layer3size), 
            b4(outputSize)
        {}

        void reset(){
            fill(input.begin(), input.end(), 0);
            fill(l2.begin(), l2.end(), 0);
            fill(l3.begin(), l3.end(), 0);
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
                    l2[i] += (w12[i][j] * input[j]); 
                }
                l2[i] += b2[i];
            }
            for (int i = 0; i < layer3size; ++i){
                for (int j = 0; j < layer2size; ++j){
                    l3[i] += (w23[i][j] * l2[j]); 
                }
                l3[i] += b3[i];
            }

            for (int i = 0; i < outputSize; ++i){
                for (int j = 0; j < layer3size; ++j){
                    output[i] += (w34[i][j] * l3[j]); 
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

int main(){

    Network net;

    ifstream map("map.txt");

    if (!map.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    string line;
    double cost;
    string path;
    vector<double> target(outputSize, 0.0);
    vector<double> output;

    Network change;
    change.fillWeights(0.0);
    double z = 0.0;

    const int batchSize = 32;
    const int numbatches = 1875;
    const double rate = 0.02;

    for (int _ = 0; _ < numbatches; ++_){
        for (int i = 0; i < batchSize; ++i){
            getline(map, line);
            path = line.substr(0, line.find(','));
            fill(target.begin(), target.end(), 0);
            target[stoi(line.substr(line.find(',')))] = 1.0;

            output = net.identify(path);
            cout << "batch " << i << ": " << net.cost(output, target) << endl;

            // first change weights & biases from layer3 to output, and store the desired changes to layer 3
            for (int j = 0; j < size(net.output); ++j){
                for (int k = 0; k < size(net.l3); ++k){
                    z += (net.w34[j][k] * net.l3[k]);
                }
                z += net.b4[j];
                for (int k = 0; k < size(change.w34[j]); ++k){
                    change.w34[j][k] += -1 * rate * net.l3[k] * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
                    change.l3[k] += net.w34[j][k] * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
                }
                change.b4[j] += -1 * rate * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
                z = 0.0;
            }

            // same thing with weights & biases from layer 2 to layer 3
            for (int j = 0; j < size(net.l3); ++j){
                for (int k = 0; k < size(net.l2); ++k){
                    z += (net.w23[j][k] * net.l2[k]);
                }
                z += net.b3[j];
                for (int k = 0; k < size(net.l2); ++k){
                    change.w23[j][k] += -1 * rate * net.l2[k] * sigmoidPrime(z) * change.l3[j] / batchSize;
                    change.l2[k] += net.w23[j][k] * change.l3[j] / batchSize;
                }
                change.b3[j] = -1 * rate * sigmoidPrime(z) * change.l3[j] / batchSize;
                z = 0.0;
            }

            // same thing with weights and biases from input to layer 1
            for (int j = 0; j < size(net.l2); ++j){
                for (int k = 0; k < size(net.input); ++k){
                    z += (net.w12[j][k] * net.input[j]);
                }
                z += net.b2[j];
                for (int k = 0; k < size(net.input); ++k){
                    change.w12[j][k] += -1 * rate * net.input[k] * sigmoidPrime(z) * change.l2[j] / batchSize;
                }
                change.b2[j] = -1 * rate * sigmoidPrime(z) * change.l2[j] / batchSize;
            }
        }

        // apply the changes
        for (int j = 0; j < net.output.size(); ++j){
            for (int k = 0; k < net.l3.size(); ++k){
                net.w34[j][k] += change.w34[j][k];
            }
            net.b4[j] += change.b4[j];
        }
        for (int j = 0; j < net.l3.size(); ++j){
            for (int k = 0; k < net.l2.size(); ++k){
                net.w23[j][k] += change.w23[j][k];
            }
            net.b3[j] += change.b3[j];
        }
        for (int j = 0; j < net.l2.size(); ++j){
            for (int k = 0; k < net.input.size(); ++k){
                net.w12[j][k] += change.w12[j][k];
            }
            net.b2[j] += change.b2[j];
        }
    }
    
    return 0;
}