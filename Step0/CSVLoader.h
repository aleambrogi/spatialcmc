#pragma once
#include "DataRow.h"
#include <string>
#include <vector>

class CSVReader {
public:
    CSVReader(const std::string& filename, char delimiter = ',');

    std::vector<DataRow> read();

private:
    std::string filename_;
    char delimiter_;
};
