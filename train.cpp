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

        void setZero(){
            output.setZero();
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
    const int batchSize = 32;
    const int numbatches = 1875;
    const double rate = 0.02;
    vector<int> sizes = {784, 183, 43, 10};

    Network net(sizes);

    ifstream map("map.txt");

    if (!map.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    string line;
    string path;
    e::VectorXd target = e::VectorXd::Zero(net.outputsize);
    e::VectorXd output;

    Network change(sizes);
    change.setZero();

    const int batchSize = 32;
    const int numbatches = 1875;
    const double rate = 0.02;

    e::VectorXd SigPrimez;

    for (int _ = 0; _ < numbatches; ++_){
        for (int i = 0; i < batchSize; ++i){
            change.setZero();
            getline(map, line);
            path = line.substr(0, line.find(','));
            target(stoi(line.substr(line.find(',')))) = 1.0;
            output = net.identify(path);

            // yes, the math was hell
            change.output = 2 * (net.output - target);
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

    return 0;
}