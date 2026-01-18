#include <iostream>
#include <vector>
#include <math.h>
#include <array>
#include <string>
#include <unistd.h>
#include <fstream>
#include <cstdint>


using namespace std;

#define inputSize 284
#define layer2size 183
#define layer3size 43
#define outputSize 10

uint32_t reverse_bytes(uint32_t bytes) {
    return (bytes >> 24) | ((bytes >> 8) & 0x0000FF00) | ((bytes << 8) & 0x00FF0000) | (bytes << 24);
}

vector<double> imageToVector(const string& imagePath, int size){
    // dumbass big endian shit

    ifstream file(imagePath, std::ios::binary);
    vector<double> normValues;

    if (!file.is_open()) {
        cerr << "Error opening file: " << imagePath << endl;
        return normValues;
    }

    uint32_t magic_number = 0;
    uint32_t num_images = 0;
    uint32_t num_rows = 0;
    uint32_t num_cols = 0;

    file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
    magic_number = reverse_bytes(magic_number);

    file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
    num_images = reverse_bytes(num_images);

    file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
    num_rows = reverse_bytes(num_rows);

    file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
    num_cols = reverse_bytes(num_cols);

    normValues.resize(size);

    for (int i = 0; i < size; ++i) {
        unsigned char pixel_value;
        file.read(reinterpret_cast<char*>(&pixel_value), sizeof(pixel_value));
        
        // Normalize the pixel value from [0, 255] to [0.0, 1.0]
        normValues[i] = static_cast<double>(pixel_value) / 255.0;
    }

    file.close();
    return normValues;

}

int main(int argc, char *argv[]){

    vector<float> input;
    vector<float> layer2;
    vector<float> layer3; 
    vector<float> output;

    array<array<float, inputSize>, layer2size> weights12; // there are (layer2size) rows of (inputSize) weights so you can matrix multiply
    array<array<float, layer2size>, layer3size> weights23; // same for these
    array<array<float, layer3size>, outputSize> weights34;

    ifstream image(argv[argc - 1]);
    if (!image){ 
        cout << "unable to open file" << endl;
        return 1;
    }

    

    return 0;
}