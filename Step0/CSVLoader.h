#pragma once
#include "DataRow.h"
#include <string>
#include <vector>

// ------------------------------------------------------------
// CSVReader: Utility to parse CSV files into DataRow objects
// ------------------------------------------------------------
class CSVReader {
public:
    CSVReader(const std::string& filename, char delimiter = ',');

    // ------------------------------------------------------------
    // Reads the file and returns a vector of DataRows
    // ------------------------------------------------------------
    std::vector<DataRow> read();

private:
    std::string filename_;
    char delimiter_;
};
