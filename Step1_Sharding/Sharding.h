#pragma once
#include <DataRow.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>

namespace Step1_Sharding {

    // ------------------------------------------------------------
    // Sharding: Manages data partitioning based on a key column
    // ------------------------------------------------------------
    class Sharding {
    public:
        using Row = DataRow;
        using Shard = std::vector<Row>;

        Sharding(const std::vector<Row>& data);

        // Splits the dataset into shards based on a key column
        std::vector<Shard> generateShards(const std::string& column);

    private:
        std::vector<Row> dataset;
    };

}
