#include "Step0/CSVLoader.h"
#include "Step1_Sharding/Sharding.h"
#include <iostream>

int main() {
    CSVReader reader("dataset.csv");
    auto data = reader.read();  // restituisce std::vector<DataRow>

    Step1_Sharding::Sharding sharder(data);
    auto shards = sharder.generateShards("provincia");

    for (size_t i = 0; i < shards.size(); ++i) {
        std::cout << "Shard " << i << ": " << shards[i].size() << " osservazioni\n";
    }

    return 0;
}

