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
#include <Eigen/Dense>

#include "helpers.h"

using namespace std;
namespace e = Eigen;

#define inputSize 784 //maybe some geometric progression going on
#define layer2size 183
#define layer3size 43
#define outputSize 10

// create a network class so it makes life easier
class Network {

    public:
        vector<int> shape;
        int layercount = shape.size();
        int outputsize = shape[layercount - 1];
        vector<e::VectorXd> l;
        vector<e::VectorXd> b;
        vector<e::MatrixXd> w;
        vector<e::VectorXd> z;
        e::VectorXd& output = l[layercount - 1];

        Network(vector<int> sizes):
            l(layercount), 
            b(layercount - 1), 
            w(layercount - 1), 
            z(layercount - 1)
        {
            shape = sizes;
            l[0] = e::VectorXd::Constant(shape[0], 0);
            for (int i = 0; i < layercount-1; ++i){
                w[i] = e::MatrixXd::Random(shape[i], shape[i+1]) * 0.5;
                b[i] = e::VectorXd::Constant(shape[i+1], 0.1);
                z[i] = e::VectorXd::Constant(shape[i+1], 0);
                l[i+1] = e::VectorXd::Constant(shape[i+1]);
            }
        }

        void reset(){
            for (int i = 0; i < layercount; ++i){
                l[i].setZero();
            }
        }

        e::VectorXd identify(const string& imagePath){
            reset();
            l[0] = imageToVector(imagePath, shape[0]);

            for (int i = 0; i < layercount-1; ++i){
                z[i] = (w[i] * l[i]) + b[i];
                l[i+1] = 1.0 / (1.0 + (-z[i]).array().exp());
            }
            return output;
        }

        double cost(e::VectorXd identification, e::VectorXd target){
            return (identification - target).array().square().sum();
        }

        void backpropogate(double cost, double stepsize){

        }
};

int main(){

    vector<int> sizes = {784, 183, 43, 10};
    Network net(sizes);

    ifstream map("map.txt");

    if (!map.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    string line;
    double cost;
    string path;
    e::VectorXd target = e::VectorXd::Zero(net.outputsize);
    e::VectorXd output;

    Network change(sizes);
    double z = 0.0;

    const int batchSize = 32;
    const int numbatches = 1875;
    const double rate = 0.02;

    for (int _ = 0; _ < numbatches; ++_){
        for (int i = 0; i < batchSize; ++i){
            getline(map, line);
            path = line.substr(0, line.find(','));
            target(stoi(line.substr(line.find(',')))) = 1.0;

            output = net.identify(path);
            cout << "batch " << i << ": " << net.cost(target, output) << endl;

            // first change weights & biases from layer3 to output, and store the desired changes to layer 3
            // for (int j = 0; j < size(net.output); ++j){

            //     for (int k = 0; k < size(change.w34[j]); ++k){
            //         change.w34[j][k] += -1 * rate * net.l3[k] * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
            //         change.l3[k] += net.w34[j][k] * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
            //     }
            //     change.b4[j] += -1 * rate * sigmoidPrime(z) * 2 * (output[j] - target[j]) / batchSize;
            //     z = 0.0;
            // }

            for (int L = net.layercount - 2; L > -1; --L){
                change.w[L] = (net.w[L] * net.l[L].asDiagonal()).array().colwise() * (net.z[L].array() * (1 - net.z[L].array()));
                if 
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