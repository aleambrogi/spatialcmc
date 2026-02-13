#include <DataRow.h>
#include "adjacency.h"

#include <fstream>
#include <stdexcept>
#include <unordered_map>

#include <nlohmann/json.hpp>
using json = nlohmann::json;

std::vector<std::vector<int>>
AdjacencyUtils::load_adjacency_json(const std::string& filename)
{
    std::ifstream f(filename);
    if (!f.is_open())
        throw std::runtime_error("Cannot open adjacency JSON file.");

    json j;
    f >> j;

    std::vector<std::vector<int>> adj(j.size());
    for (int i = 0; i < j.size(); i++)
        for (auto v : j[i])
            adj[i].push_back(v);

    return adj;
}

// --------------------------------------------------

std::vector<std::vector<int>>
AdjacencyUtils::extract_local_adjacency(
    const std::vector<DataRow>& shard,
    const std::vector<std::vector<int>>& global_adj)
{
    int n = shard.size();

    std::unordered_map<int,int> map_global_to_local;
    for (int i = 0; i < n; i++)
        map_global_to_local[ shard[i].global_index ] = i;

    std::vector<std::vector<int>> local_adj(n);

    for (int i = 0; i < n; i++)
    {
        int gi = shard[i].global_index;
        for (int g_nei : global_adj[gi])
            if (map_global_to_local.count(g_nei))
                local_adj[i].push_back(map_global_to_local[g_nei]);
    }

    return local_adj;
}

