#include <iostream>
#include <vector>
#include <math.h>
#include <cmath>
#include <array>
#include <string>
#include <unistd.h>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <random>
#include <chrono>
#include <filesystem>
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
        int layercount;
        int outputsize;
        vector<e::VectorXd> l;
        vector<e::VectorXd> b;
        vector<e::MatrixXd> w;
        vector<e::VectorXd> z;

        Network(vector<int> sizes)
        {
            shape = sizes;
            layercount = sizes.size();
            outputsize = shape[layercount - 1];
            l.push_back(e::VectorXd::Constant(shape[0], 0));
            for (int i = 0; i < layercount-1; ++i){
                w.push_back(e::MatrixXd::Random(shape[i+1], shape[i]) * 0.5);
                b.push_back(e::VectorXd::Constant(shape[i+1], 0.1));
                z.push_back(e::VectorXd::Constant(shape[i+1], 0));
                l.push_back(e::VectorXd::Constant(shape[i+1], 0));
            }
        }

        void reset(){
            for (int i = 0; i < layercount; ++i){
                l[i].setZero();
            }
        }

        e::VectorXd identify(e::VectorXd pixels){
            reset();
            l[0] = pixels;

            for (int i = 0; i < layercount-1; ++i){
                z[i] = (w[i] * l[i]) + b[i];
                l[i+1] = 1.0 / (1.0 + (-z[i]).array().exp());
            }
            return l.back();
        }

        double cost(e::VectorXd identification, e::VectorXd target){
            return (identification - target).array().square().sum();
        }

        void setZero(){
            l.back().setZero();
            for (int i = 0; i < layercount - 1; i++){
                l[i].setZero();
                b[i].setZero();
                w[i].setZero();
                z[i].setZero();
            }
        }

        void backpropogate(double cost, double stepsize){

        }
};

int main(){
    const int maxEpochs = 1000;
    const int batchSize = 160;
    const int numbatches = 375;
    const double rate = 2.0;

    vector<int> sizes = {784, 183, 43, 10};
    vector<pair<e::VectorXd, int>> cases;
    vector<pair<e::VectorXd, int>> tests;
    e::VectorXd row(sizes[0]);
    double label;
    string cell;

    string line;

    Network net(sizes);

    ifstream casesFile("./archive/mnist_train.csv");
    ifstream testsFile("./archive/mnist_test.csv");
    ofstream weights("weights.txt");

    if (!casesFile.is_open()){
        cerr << "error opening mnist_train.csv" << endl;
        return 1;
    }

    if (!testsFile.is_open()){
        cerr << "error opening mnist_test.csv" << endl;
        return 1;
    }

    if (!weights.is_open()){
        cerr << "error opening weights.txt" << endl;
        return 1;
    }
    // store the tests/cases so we can easily access them later
    getline(casesFile, line);
    while (getline(casesFile, line)){
        int i = 0;
        stringstream ss(line);

        getline(ss, cell, ',');
        label = stod(cell);

        while (getline(ss, cell, ',')){
            row[i] = stod(cell) / 255.0;
            ++i;
        }

        cases.push_back({row, label});
    }

    getline(testsFile, line);
    while (getline(testsFile, line)){
        int i = 0;
        stringstream ss(line);

        getline(ss, cell, ',');
        label = stod(cell);

        while (getline(ss, cell, ',')){
            row[i] = stod(cell) / 255.0;
            ++i;
        }

        tests.push_back({row, label});
    }

    string path;
    e::VectorXd target = e::VectorXd::Zero(net.outputsize);

    Network change(sizes);
    change.setZero();
    
    double percent = 0.0;

    e::VectorXd SigPrimez;
    e::Index maxVal;

    for (int e = 0; e < maxEpochs; e++){
        for (int i = 0; i < numbatches; ++i){
            change.setZero();
            for (int j = 0; j < batchSize; ++j){
                target.setZero();
                target(cases[batchSize*i + j].second) = 1.0;
                net.identify(cases[batchSize*i + j].first); // necessary to update values of layers

                // yes, the math was hell
                change.l.back() = 2 * (net.l.back() - target);
                for (int L = net.layercount - 2; L > -1; --L){
                    SigPrimez = (net.l[L+1].array() * (1 - net.l[L+1].array())).matrix();
                    change.w[L] += (change.l[L+1].array() * SigPrimez.array()).matrix() * net.l[L].transpose();
                    change.b[L] += (SigPrimez.array() * change.l[L+1].array()).matrix();
                    change.l[L] = net.w[L].transpose() * (SigPrimez.array() * change.l[L+1].array()).matrix();
                }
            }

            for (int L = 0; L < net.layercount - 1; L++){
                net.w[L] += -1 * rate * change.w[L] / batchSize;
                net.b[L] += -1 * rate * change.b[L] / batchSize;
            }

            if ((i + 1) % 50 == 0) cout << "  Batch " << (i + 1) << " / " << numbatches << "\r";
        }
        for (auto& entry : tests){
            net.identify(entry.first).maxCoeff(&maxVal);
            if (maxVal == entry.second){ percent += 1; };
        }

        std::cout << "\n----- EPOCH " << e << " -----" << endl;
        std::cout << "   correct: " << percent << endl;
        cout << "   percent: " << (percent / tests.size()) * 100 << "%" << endl;
        
        if ((e + 1) % 2 == 0) {
            for (int L = 0; L < net.layercount - 1; ++L){
                filesystem::resize_file("weights.txt", 0);
                weights << "-------- LAYER " << L << " EPOCH " << e << " --------" << endl;
                weights << net.w[L] << endl << endl;
                weights << net.b[L] << endl << endl;
            }
        }

        percent = 0;
    }

    return 0;
}