#pragma once
#include <vector>
#include <string>

class Sharding {
public:
    Sharding() = default;

    // Carica i dati da file CSV o altro formato
    bool loadData(const std::string& filename);

    // Divide il dataset in "numShards" shard
    void splitShards(int numShards);

    // Getter dei dati di ogni shard (opzionale)
    std::vector<std::vector<std::vector<double>>> getShards() const;

private:
    std::vector<std::vector<double>> dataset;  // dati originali
    std::vector<std::vector<std::vector<double>>> shards; // dati divisi in shard
};
