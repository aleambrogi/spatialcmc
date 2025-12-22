#pragma once
#include <string>

struct DataRow {
    std::string COMUNE;
    std::string PROVINCIA;
    double median_value;

    int global_index;

    DataRow() : median_value(0.0), global_index(-1) {}
};
