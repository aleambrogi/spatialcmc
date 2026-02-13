#ifndef ADJACENCY_H
#define ADJACENCY_H

#include <string>
#include <vector>
#include "DataRow.h"

class AdjacencyUtils
{
public:
    static std::vector<std::vector<int>>
    load_adjacency_json(const std::string& filename);

    static std::vector<std::vector<int>>
    extract_local_adjacency(
        const std::vector<DataRow>& shard,
        const std::vector<std::vector<int>>& global_adj);
};

#endif