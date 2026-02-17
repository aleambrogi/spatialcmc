#include "Sharding.h"
#include <algorithm>
#include <cctype>
#include <iostream>

namespace Step1_Sharding {

static inline std::string trim_quotes_and_ws(const std::string &s) {
    size_t i = 0, j = s.size();
    // trim whitespace
    while (i < j && std::isspace((unsigned char)s[i])) ++i;
    while (j > i && std::isspace((unsigned char)s[j-1])) --j;
    // remove surrounding double quotes if present
    if (j - i >= 2 && s[i] == '"' && s[j-1] == '"') { ++i; --j; }
    return s.substr(i, j - i);
}

Sharding::Sharding(const std::vector<Row>& data) : dataset(data) {}

// ------------------------------------------------------------
// Splits dataset into shards based on the specified column
// ------------------------------------------------------------
std::vector<Sharding::Shard> Sharding::generateShards(const std::string& column) {
    if (dataset.empty()) {
        std::cerr << "Empty dataset.\n";
        return {};
    }

    std::string col = column;
    // remove potential quotes in the passed column name
    col = trim_quotes_and_ws(col);

    std::set<std::string> uniqueValues;
    uniqueValues.clear();

    // collect some examples for debug
    int debug_print = 10;
    std::cout << "Sharding by column: \"" << col << "\" (showing up to " << debug_print << " sample values)\n";

    for (size_t i = 0; i < dataset.size(); ++i) {
        const auto &row = dataset[i];

        std::string val;
        if (col == "PROVINCIA") {
            val = trim_quotes_and_ws(row.PROVINCIA);
        } else if (col == "COMUNE") {
            val = trim_quotes_and_ws(row.COMUNE);
        } else {
            // fallback: try to use a generic field map if present (backward compat)
            #ifdef HAS_FIELD_MAP
            try {
                val = trim_quotes_and_ws(row.fields.at(col));
            } catch(...) {
                val = "";
            }
            #else
            // unknown column name => throw so user notices
            throw std::runtime_error("Sharding::generateShards: unrecognized column: " + col);
            #endif
        }

        uniqueValues.insert(val);

        if (i < (size_t)debug_print)
            std::cout << "  row[" << i << "] -> \"" << val << "\"\n";
    }

    std::cout << "Unique shard keys found: " << uniqueValues.size() << "\n";

    // map value -> shard index
    std::unordered_map<std::string, size_t> indexMap;
    size_t idx = 0;
    for (const auto &v : uniqueValues) {
        indexMap[v] = idx++;
    }

    // ------------------------------------------------------------
    // Distribute rows into shards
    // ------------------------------------------------------------
    std::vector<Shard> shards(uniqueValues.size());
    for (const auto &row : dataset) {
        std::string val;
        if (col == "PROVINCIA") val = trim_quotes_and_ws(row.PROVINCIA);
        else if (col == "COMUNE") val = trim_quotes_and_ws(row.COMUNE);
        else val = "";

        shards[indexMap[val]].push_back(row);
    }

    std::cout << "Number of generated shards: " << shards.size() << std::endl;
    return shards;
}

}// namespace
