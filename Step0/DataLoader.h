#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <string>
#include <vector>
#include "DataRow.h"

enum class DataFormat { CSV, SHP, GPKG };

struct LoadSpec {
    std::string path;
    DataFormat format;

    // Attribute field names for GIS inputs
    std::string provincia_field = "PROVINCIA";
    std::string comune_field = "COMUNE";
    std::string value_field = "median_val"; // fallback to "Value" if not found

    // For GPKG you may optionally specify a layer name (empty = first layer)
    std::string layer_name = "";
};

std::vector<DataRow> load_dataset(const LoadSpec& spec);

#endif
