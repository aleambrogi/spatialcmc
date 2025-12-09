#include "CSVLoader.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

CSVReader::CSVReader(const std::string& filename, char delimiter)
    : filename_(filename), delimiter_(delimiter) {}

std::string strip_quotes(const std::string& s) {
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}
std::vector<DataRow> CSVReader::read() {
    std::ifstream file(filename_);
    if (!file.is_open()) {
        throw std::runtime_error("Impossibile aprire il file CSV: " + filename_);
    }

    std::string line;

    // --- Leggi header ---
    if (!std::getline(file, line)) {
        throw std::runtime_error("CSV vuoto o header mancante.");
    }

    std::stringstream headerStream(line);
    std::vector<std::string> columns;
    std::string col;
    while (std::getline(headerStream, col, delimiter_)) {
        columns.push_back(strip_quotes(col));  // rimuove i doppi apici
    }

    // Trova gli indici delle colonne importanti
    int idx_comune = -1, idx_provincia = -1, idx_median = -1;
    for (size_t i = 0; i < columns.size(); ++i) {
        if (columns[i] == "COMUNE") idx_comune = i;
        else if (columns[i] == "PROVINCIA") idx_provincia = i;
        else if (columns[i] == "median_value") idx_median = i;
    }
    if (idx_comune == -1 || idx_provincia == -1 || idx_median == -1) {
        throw std::runtime_error("Colonne COMUNE, PROVINCIA o median_value mancanti nel CSV.");
    }

    // --- Leggi tutte le righe ---
    std::vector<DataRow> rows;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string cell;
        std::vector<std::string> tokens;
        while (std::getline(ss, cell, delimiter_)) {
            tokens.push_back(cell);
        }

        DataRow row;
        row.COMUNE = tokens[idx_comune];
        row.PROVINCIA = tokens[idx_provincia];

        try {
            row.median_value = std::stod(tokens[idx_median]);
        } catch (...) {
            row.median_value = 0.0; // o NaN se vuoi usare <cmath>
        }

        rows.push_back(row);
    }

    return rows;
}
