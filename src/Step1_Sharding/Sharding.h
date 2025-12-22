#pragma once
#include "../Step0/DataRow.h"
#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>

namespace Step1_Sharding {

    class Sharding {
    public:
        using Row = DataRow;
        using Shard = std::vector<Row>;

        Sharding(const std::vector<Row>& data);

        // Divide il dataset in shard basati su una colonna chiave
        std::vector<Shard> generateShards(const std::string& column);

    private:
        std::vector<Row> dataset;
    };

}
