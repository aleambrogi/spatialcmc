#ifndef GEOMETRY_ADJACENCY_H
#define GEOMETRY_ADJACENCY_H

#include <vector>
#include "DataRow.h"

namespace Step0 {

    enum class AdjacencyMode {
        None,
        KNN,
        PolygonTouches,
        PolygonIntersects
    };

    AdjacencyMode infer_adjacency_mode(const std::vector<DataRow>& shard);

    std::vector<std::vector<int>>
    build_local_adjacency_from_rows(const std::vector<DataRow>& shard,
                                   AdjacencyMode mode,
                                   int knn_k = 8);

} // namespace Step0

#endif
