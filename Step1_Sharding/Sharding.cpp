#include "Sharding.h"
#include <fstream>
#include <sstream>
#include <iostream>

bool Sharding::loadData(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore apertura file: " << filename << std::endl;
        return false;
    }

    dataset.clear();
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<double> row;
        std::string value;
        while (std::getline(ss, value, ',')) {
            row.push_back(std::stod(value));
        }
        dataset.push_back(row);
    }

    file.close();
    return true;
}

void Sharding::splitShards(int numShards) {
    shards.clear();
    shards.resize(numShards);

    for (size_t i = 0; i < dataset.size(); ++i) {
        shards[i % numShards].push_back(dataset[i]);
    }

    std::cout << "Dataset diviso in " << numShards << " shard." << std::endl;
}

std::vector<std::vector<std::vector<double>>> Sharding::getShards() const {
    return shards;
}
