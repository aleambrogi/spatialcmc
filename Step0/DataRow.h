#pragma once
#include <string>

struct DataRow {
    std::string COMUNE;
    std::string PROVINCIA;
    double median_value;

    DataRow() : median_value(0.0) {}
};
