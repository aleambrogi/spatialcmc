#include "CSVLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

CSVReader::CSVReader(const std::string& filename, char delimiter)
    : filename_(filename), delimiter_(delimiter) {}

std::vector<DataRow> CSVReader::read() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire il file CSV: " + filename_);
    }

    std::string line;
    std::vector<DataRow> rows;

    // --- Leggi header ---
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV vuoto o header mancante.");
    }

    std::stringstream headerStream(line);
    std::vector<std::string> columns;
    std::string col;

    while (std::getline(headerStream, col, delimiter_)) {
        columns.push_back(col);
    }

    // --- Leggi tutte le righe ---
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        DataRow row;

        int i = 0;
        while (std::getline(ss, cell, delimiter_)) {
            if (i < (int)columns.size()) {
                row.fields[columns[i]] = cell;
            }
            i++;
        }

        rows.push_back(row);
    }

    return rows;
}
