#include <iostream>
#include <vector>
#include <math.h>
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

int main(int argc, char *argv[]){
    // random stuff
    unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 gen(seed);
    uniform_int_distribution<int> distrib(-0.5, 0.5);

    // intialize weights and biases
    vector<double> input;
    vector<double> baises2(layer2size, 0.0);
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

    ifstream map("map.txt");

    if (!map.is_open()){
        cerr << "error opening map.txt" << endl;
        return 1;
    }

    string line;

    while (getline(map, line)){
        string path = line.substr(0, line.find(','));
        int target = stoi(line.substr(line.find(',')));
    }
    

    return 0;
}