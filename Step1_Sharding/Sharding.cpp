#include "Sharding.h"

namespace Step1_Sharding {

    Sharding::Sharding(const std::vector<Row>& data) : dataset(data) {}

    std::vector<Sharding::Shard> Sharding::generateShards(const std::string& column) {
        if (dataset.empty()) {
            std::cerr << "Dataset vuoto.\n";
            return {};
        }

        // Trova valori unici della colonna
        std::set<std::string> uniqueValues;
        for (const auto& row : dataset) {
            uniqueValues.insert(row.fields.at(column));
        }

        // Mappa valore -> indice shard
        std::unordered_map<std::string, size_t> indexMap;
        size_t idx = 0;
        for (const auto& val : uniqueValues)
            indexMap[val] = idx++;

        // Crea shard
        std::vector<Shard> shards(uniqueValues.size());
        for (const auto& row : dataset) {
            shards[indexMap[row.fields.at(column)]].push_back(row);
        }

        std::cout << "Numero di shard generati: " << shards.size() << std::endl;
        return shards;
    }

}
