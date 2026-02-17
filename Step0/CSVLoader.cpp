#include "CSVLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

CSVReader::CSVReader(const std::string& filename, char delimiter)
    : filename_(filename), delimiter_(delimiter) {}

static inline std::string trim_quotes_ws(const std::string &s) {
    size_t i = 0, j = s.size();
    while (i < j && isspace((unsigned char)s[i])) ++i;
    while (j > i && isspace((unsigned char)s[j-1])) --j;
    if (j - i >= 2 && s[i] == '"' && s[j-1] == '"') { ++i; --j; }
    return s.substr(i, j - i);
}

static inline double safe_stod(const std::string& s) {
    std::string t;
    for (char c : s)
        if (!isspace((unsigned char)c)) t.push_back(c);

    if (t.empty()) return 0.0;
    if (t == "NA" || t == "Na" || t == "na" || t == "N/A") return 0.0;

    try {
        return std::stod(t);
    } catch(...) {
        std::cerr << "WARN: unable to convert string \""
                  << s << "\" to double -> setting median_val = 0\n";
        return 0.0;
    }
}

std::vector<DataRow> CSVReader::read() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Unable to open CSV file: " + filename_);
    }

    std::string line;
    std::vector<DataRow> rows;

    // ------------------------------------------------------------
    // Read header
    // ------------------------------------------------------------
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV or missing header.");
    }

    std::stringstream headerStream(line);
    std::vector<std::string> columns;
    std::string col;

    while (std::getline(headerStream, col, delimiter_)) {
        columns.push_back(trim_quotes_ws(col));
    }

    // ------------------------------------------------------------
    // Read all rows
    // ------------------------------------------------------------
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;

        DataRow row;

        int i = 0;
        while (std::getline(ss, cell, delimiter_)) {
            std::string clean = trim_quotes_ws(cell);
            if (i < (int)columns.size()) {
                const std::string &cname = columns[i];

                // Populate DataRow struct (assuming COMUNE, PROVINCIA, median_val)
                if (cname == "COMUNE") row.COMUNE = clean;
                else if (cname == "PROVINCIA") row.PROVINCIA = clean;
                else if (cname == "median_val")
                    row.median_val = safe_stod(clean);
                else {
                    // Expand here for extra fields
                }
            }
            i++;
        }

        rows.push_back(row);
    }

    return rows;
}