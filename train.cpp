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

using namespace std;
namespace e = Eigen;


// create a network class so it makes life easier
class Network {

    public:
        vector<int> shape;
        int layercount;
        int outputsize;
        vector<e::MatrixXd> l;
        vector<e::VectorXd> b;
        vector<e::MatrixXd> w;
        vector<e::MatrixXd> z;

        Network(vector<int> sizes)
        {
            shape = sizes;
            layercount = sizes.size();
            outputsize = shape[layercount - 1];
            l.push_back(e::MatrixXd::Constant(shape[0], 1, 0.0));
            for (int i = 0; i < layercount-1; ++i){
                double scale = sqrt(1.0 / shape[i]);
                w.push_back(e::MatrixXd::Random(shape[i+1], shape[i]) * scale);
                b.push_back(e::VectorXd::Constant(shape[i+1], 0.1));
                z.push_back(e::MatrixXd::Constant(shape[i+1], 1, 0.0));
                l.push_back(e::MatrixXd::Constant(shape[i+1], 1, 0.0));
            }
        }

        void reset(){
            for (int i = 0; i < layercount; ++i){
                l[i].setZero();
            }
        }

        e::MatrixXd identify(const e::MatrixXd& input){
            reset();
            l[0] = input;

            for (int i = 0; i < layercount-1; ++i){
                z[i] = (w[i] * l[i]).colwise() + b[i];
                l[i+1] = 1.0 / (1.0 + (-z[i]).array().exp());
            }
            return l.back();
        }

        e::VectorXd identify(const e::VectorXd& input){
            e::MatrixXd m(input.size(), 1);
            m.col(0) = input;
            return identify(m).col(0);
        }

        double cost(const e::MatrixXd& identification, const e::MatrixXd& target){
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
};

int main(){
    const int maxEpochs = 100;
    const int batchSize = 480;
    const int numbatches = 125;
    const static e::IOFormat CSVFormat(e::StreamPrecision, e::DontAlignCols, ", ", "\n");
    double rate = 2.5;

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

    if (!casesFile.is_open()){
        cerr << "error opening mnist_train.csv" << endl;
        return 1;
    }

    if (!testsFile.is_open()){
        cerr << "error opening mnist_test.csv" << endl;
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

    random_device rd;
    mt19937 g(rd());
    shuffle(cases.begin(), cases.end(), g);

    string path;
    e::MatrixXd target = e::MatrixXd::Zero(net.outputsize, batchSize);
    e::MatrixXd input(sizes[0], batchSize);

    Network change(sizes);
    change.setZero();
    
    double percent = 0.0;

    e::MatrixXd SigPrimez;
    e::Index maxVal;

    for (int e = 0; e < maxEpochs; e++){
        for (int i = 0; i < numbatches; ++i){
            change.setZero();
            target.setZero();
            for (int j = 0; j < batchSize; ++j){
                target(cases[batchSize*i + j].second, j) = 1.0;
                input.col(j) = cases[batchSize*i + j].first;
            }

            net.identify(input); // necessary to update values of layers

            // yes, the math was hell
            change.l.back() = 2 * (net.l.back() - target);
            for (int L = net.layercount - 2; L > -1; --L){
                SigPrimez = (net.l[L+1].array() * (1 - net.l[L+1].array())).matrix();
                change.w[L] += (change.l[L+1].array() * SigPrimez.array()).matrix() * net.l[L].transpose();
                change.b[L] += (SigPrimez.array() * change.l[L+1].array()).matrix().rowwise().sum();
                change.l[L] = net.w[L].transpose() * (SigPrimez.array() * change.l[L+1].array()).matrix();
            }
            
            for (int L = 0; L < net.layercount - 1; L++){
                net.w[L] += -1 * rate * change.w[L] / batchSize;
                net.b[L] += -1 * rate * change.b[L] / batchSize;
            }

            // if ((i + 1) % 50 == 0) cout << "  Batch " << (i + 1) << " / " << numbatches << "\r";
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
                ofstream weights("weights/weights" + to_string(L) + ".csv", ios::trunc);
                ofstream biases("weights/biases" + to_string(L) + ".csv", ios::trunc);
                if (!weights.is_open()){
                    cerr << "error opening weights" + to_string(L) + ".csv" << endl;
                    return 1;
                }
                if (!biases.is_open()){
                    cerr << "error opening biases" + to_string(L) + ".csv" << endl;
                    return 1;
                }
                weights << std::fixed;
                weights.precision(7);
                biases << std::fixed;
                biases.precision(7);
                weights << net.w[L].format(CSVFormat);
                biases << net.b[L].format(CSVFormat);
                weights.close();
                biases.close();
            }
        }

        percent = 0;
    }

    return 0;
}