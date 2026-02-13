#pragma once
#include <string>

struct DataRow {
    std::string COMUNE;
    std::string PROVINCIA;
    double median_val;

    int global_index;

    DataRow() : median_val(0.0), global_index(-1) {}

    std::string geometry_wkt;
};
