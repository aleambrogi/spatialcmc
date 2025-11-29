#pragma once
#include <string>
#include <unordered_map>

struct DataRow {
    // Chiave = nome colonna, Valore = contenuto cella
    std::unordered_map<std::string, std::string> fields;

    const std::string& operator[](const std::string& key) const {
        return fields.at(key);
    }

    std::string& operator[](const std::string& key) {
        return fields[key];
    }
};
