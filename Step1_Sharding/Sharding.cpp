#include "Sharding.h"
#include <iostream>
#include <unordered_map>
#include <set>

namespace Step1_Sharding {

    Sharding::Sharding(const std::vector<Row>& data) : dataset(data) {}

    std::vector<Sharding::Shard> Sharding::generateShards(const std::string& column) {
        if (dataset.empty()) {
            std::cerr << "Dataset vuoto.\n";
            return {};
        }

        // --- Trova valori unici della colonna scelta ---
        std::set<std::string> uniqueValues;
        for (const auto& row : dataset) {
            std::string key;
            if (column == "PROVINCIA") key = row.PROVINCIA;
            else if (column == "COMUNE") key = row.COMUNE;
            else throw std::runtime_error("Colonna non supportata per lo sharding: " + column);

            uniqueValues.insert(key);
        }

        // --- Mappa valore -> indice shard ---
        std::unordered_map<std::string, size_t> indexMap;
        size_t idx = 0;
        for (const auto& val : uniqueValues)
            indexMap[val] = idx++;

        // --- Crea shard ---
        std::vector<Shard> shards(uniqueValues.size());
        for (const auto& row : dataset) {
            std::string key;
            if (column == "PROVINCIA") key = row.PROVINCIA;
            else key = row.COMUNE;

            shards[indexMap[key]].push_back(row);
        }

        std::cout << "Numero di shard generati: " << shards.size() << std::endl;
        return shards;
    }

}
