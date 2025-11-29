#pragma once
#include <string>
#include <vector>
#include "DataRow.h"

class CSVReader {
public:
    CSVReader(const std::string& filename, char delimiter = ',');

    // Legge il CSV e restituisce un vettore di DataRow
    std::vector<DataRow> read();

private:
    std::string filename_;
    char delimiter_;
};
