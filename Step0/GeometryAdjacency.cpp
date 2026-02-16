#include "GeometryAdjacency.h"

#include <ogrsf_frmts.h>
#include <algorithm>
#include <cmath>

namespace {

// Helpers in anonymous namespace (internal linkage)
struct GeomCache {
    OGRGeometry* geom = nullptr;
    OGREnvelope env;
    double cx = 0.0, cy = 0.0;
};

static OGRGeometry* geom_from_wkt(const std::string& wkt) {
    if (wkt.empty()) return nullptr;
    OGRGeometry* g = nullptr;
    char* wkt_c = const_cast<char*>(wkt.c_str());
    if (OGRGeometryFactory::createFromWkt(&wkt_c, nullptr, &g) != OGRERR_NONE) {
        return nullptr;
    }
    return g;
}

static bool compute_centroid_xy(OGRGeometry* g, double& x, double& y) {
    if (!g) return false;
    OGRPoint p;
    if (g->Centroid(&p) != OGRERR_NONE) return false;
    x = p.getX();
    y = p.getY();
    return true;
}

static double sqdist(double x1,double y1,double x2,double y2){
    double dx=x1-x2, dy=y1-y2;
    return dx*dx + dy*dy;
}

static bool bbox_overlaps(const OGREnvelope& a, const OGREnvelope& b) {
    return !(a.MaxX < b.MinX || b.MaxX < a.MinX || a.MaxY < b.MinY || b.MaxY < a.MinY);
}

static std::vector<std::vector<int>>
build_knn(const std::vector<GeomCache>& cache, int k)
{
    const int n = (int)cache.size();
    std::vector<std::vector<int>> adj(n);
    if (n <= 1) return adj;

    k = std::max(1, std::min(k, n - 1));

    for (int i = 0; i < n; ++i) {
        std::vector<std::pair<double,int>> d;
        d.reserve(n - 1);
        for (int j = 0; j < n; ++j) if (j != i) {
            d.emplace_back(sqdist(cache[i].cx, cache[i].cy, cache[j].cx, cache[j].cy), j);
        }

        std::nth_element(d.begin(), d.begin() + k, d.end(),
                         [](const auto& a, const auto& b){ return a.first < b.first; });

        adj[i].reserve(k);
        for (int t = 0; t < k; ++t) adj[i].push_back(d[t].second);
    }

    // Undirected closure
    for (int i = 0; i < n; ++i) {
        for (int j : adj[i]) {
            if (std::find(adj[j].begin(), adj[j].end(), i) == adj[j].end())
                adj[j].push_back(i);
        }
    }
    return adj;
}

static std::vector<std::vector<int>>
build_poly_adj(const std::vector<GeomCache>& cache, bool use_intersects)
{
    const int n = (int)cache.size();
    std::vector<std::vector<int>> adj(n);

    for (int i = 0; i < n; ++i) {
        if (!cache[i].geom) continue;
        for (int j = i + 1; j < n; ++j) {
            if (!cache[j].geom) continue;

            if (!bbox_overlaps(cache[i].env, cache[j].env)) continue;

            bool ok = use_intersects
                    ? cache[i].geom->Intersects(cache[j].geom)
                    : cache[i].geom->Touches(cache[j].geom);

            if (ok) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }
    return adj;
}

} // anonymous namespace

namespace Step0 {

AdjacencyMode infer_adjacency_mode(const std::vector<DataRow>& shard)
{
    GDALAllRegister();

    for (const auto& r : shard) {
        OGRGeometry* g = geom_from_wkt(r.geometry_wkt);
        if (!g) continue;

        auto t = wkbFlatten(g->getGeometryType());
        OGRGeometryFactory::destroyGeometry(g);

        if (t == wkbPoint || t == wkbMultiPoint) return AdjacencyMode::KNN;
        if (t == wkbPolygon || t == wkbMultiPolygon) return AdjacencyMode::PolygonIntersects;
    }
    return AdjacencyMode::None;
}

std::vector<std::vector<int>>
build_local_adjacency_from_rows(const std::vector<DataRow>& shard,
                               AdjacencyMode mode,
                               int knn_k)
{
    GDALAllRegister();

    const int n = (int)shard.size();
    std::vector<GeomCache> cache(n);

    for (int i = 0; i < n; ++i) {
        cache[i].geom = geom_from_wkt(shard[i].geometry_wkt);
        if (cache[i].geom) {
            cache[i].geom->getEnvelope(&cache[i].env);
            compute_centroid_xy(cache[i].geom, cache[i].cx, cache[i].cy);
        }
    }

    std::vector<std::vector<int>> adj;
    switch (mode) {
        case AdjacencyMode::None:
            adj.assign(n, {});
            break;
        case AdjacencyMode::KNN:
            adj = build_knn(cache, knn_k);
            break;
        case AdjacencyMode::PolygonTouches:
            adj = build_poly_adj(cache, false);
            break;
        case AdjacencyMode::PolygonIntersects:
            adj = build_poly_adj(cache, true);
            break;
        default:
            adj.assign(n, {});
            break;
    }

    for (auto& c : cache) {
        if (c.geom) OGRGeometryFactory::destroyGeometry(c.geom);
        c.geom = nullptr;
    }

    return adj;
}

} // namespace Step0