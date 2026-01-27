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
                w.push_back(e::MatrixXd::Random(shape[i], shape[i+1]) * 0.5);
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

        e::VectorXd identify(const string& imagePath){
            reset();
            l[0] = imageToVector(imagePath, shape[0]);

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
    const int batchSize = 80;
    const int numbatches = 750;
    const double rate = 0.02;

    vector<int> sizes = {784, 183, 43, 10};
    vector<pair<string, int>> cases;
    vector<int> correct;
    vector<pair<string, int>> tests;

    string line;

    Network net(sizes);

    ifstream casesFile("map.txt");
    ifstream testsFile("tests.txt");
    ofstream weights("weights.txt");

    if (!casesFile.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    if (!testsFile.is_open()){
        cerr << "error opening tests.txt" << endl;
        return 1;
    }

    if (!weights.is_open()){
        cerr << "error opening weights.txt" << endl;
        return 1;
    }
    // store the tests/cases so we can easily access them later
    while (getline(casesFile, line)){
        cases.push_back({line.substr(0, line.find(',')), stoi(line.substr(line.find(',')))});
    }

    while (getline(testsFile, line)){
        tests.push_back({line.substr(0, line.find(',')), stoi(line.substr(line.find(',')))});
    }

    string path;
    e::VectorXd target = e::VectorXd::Zero(net.outputsize);
    e::VectorXd output;

    Network change(sizes);
    change.setZero();
    
    double percent = 0.0;

    e::VectorXd SigPrimez;

    for (int e = 0; e < maxEpochs; e++){
        for (int i = 0; i < numbatches; ++i){
            for (int j = 0; j < batchSize; ++j){
                change.setZero();
                target.setZero();
                path = cases[batchSize*i + j].first;
                target(cases[batchSize*i + j].second) = 1.0;
                net.identify(path); // necessary to update values of layers

                // yes, the math was hell
                change.l.back() = 2 * (net.l.back() - target);
                for (int L = net.layercount - 2; L > -1; --L){
                    SigPrimez = (net.z[L].array() * (1 - net.z[L].array())).matrix();
                    change.w[L] += ((net.w[L] * net.l[L].asDiagonal()).array().colwise() * SigPrimez.array() * change.l[L+1].array()).matrix() / batchSize;
                    change.b[L] += (SigPrimez.array() * change.l[L+1].array()).matrix() / batchSize;
                    change.l[L] += net.w[L].transpose() * (SigPrimez.array() * change.l[L+1].array()).matrix() / batchSize;
                }
            }

            for (int L = 0; L < net.layercount-1; ++L){
                net.w[L] += -1 * rate * change.w[L];
                net.b[L] += -1 * rate * change.b[L];
            }
        }
        for (auto& entry : tests){
            if (net.identify(entry.first).maxCoeff() == entry.second){ percent += 1; };
        }

        cout << "----- EPOCH " << e << " -----" << endl;
        cout << "   percent: " << percent/tests.size() << endl << endl;
        for (int L = 0; L < net.layercount - 2; ++L){
            weights << "-------- LAYER " << L << " --------" << endl;
            weights << net.w[L] << endl << endl;
            weights << net.b[L] << endl << endl;
        }

    }

    return 0;
}