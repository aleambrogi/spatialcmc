#include "DataLoader.h"
#include "CSVLoader.h"   // your CSVReader

#include <stdexcept>
#include <iostream>
#include <algorithm>

// GDAL/OGR includes
#include <ogrsf_frmts.h>

static bool has_field(OGRFeatureDefn* defn, const std::string& name) {
    return defn->GetFieldIndex(name.c_str()) >= 0;
}

static std::string get_string_field(OGRFeature* f, OGRFeatureDefn* defn, const std::string& name) {
    int idx = defn->GetFieldIndex(name.c_str());
    if (idx < 0) return "";
    if (!f->IsFieldSetAndNotNull(idx)) return "";
    return f->GetFieldAsString(idx);
}

static double get_double_field(OGRFeature* f, OGRFeatureDefn* defn,
                               const std::string& preferred, const std::string& fallback)
{
    int idx = defn->GetFieldIndex(preferred.c_str());
    if (idx < 0) idx = defn->GetFieldIndex(fallback.c_str());
    if (idx < 0) {
        throw std::runtime_error("Value field not found: '" + preferred + "' (or fallback '" + fallback + "')");
    }
    if (!f->IsFieldSetAndNotNull(idx)) return 0.0;
    return f->GetFieldAsDouble(idx);
}

static std::vector<DataRow> load_csv_rows(const std::string& path) {
    CSVReader reader(path, ',');
    return reader.read();
}

static std::vector<DataRow> load_vector_rows_ogr(const LoadSpec& spec) {
    GDALAllRegister();

    GDALDataset* ds = (GDALDataset*) GDALOpenEx(
        spec.path.c_str(),
        GDAL_OF_VECTOR,
        nullptr, nullptr, nullptr
    );
    if (!ds) {
        throw std::runtime_error("GDALOpenEx failed for: " + spec.path);
    }

    OGRLayer* layer = nullptr;
    if (!spec.layer_name.empty()) {
        layer = ds->GetLayerByName(spec.layer_name.c_str());
        if (!layer) {
            GDALClose(ds);
            throw std::runtime_error("Layer not found in dataset: " + spec.layer_name);
        }
    } else {
        layer = ds->GetLayer(0);
        if (!layer) {
            GDALClose(ds);
            throw std::runtime_error("No layers found in dataset.");
        }
    }

    OGRFeatureDefn* defn = layer->GetLayerDefn();
    if (!defn) {
        GDALClose(ds);
        throw std::runtime_error("Layer definition not available.");
    }

    // Basic field checks (don’t hard fail for provincia/comune, but value must exist)
    const bool has_prov = has_field(defn, spec.provincia_field);
    const bool has_comu = has_field(defn, spec.comune_field);
    if (!has_prov) std::cerr << "[WARN] Missing field '" << spec.provincia_field << "'.\n";
    if (!has_comu) std::cerr << "[WARN] Missing field '" << spec.comune_field << "'.\n";

    layer->ResetReading();

    std::vector<DataRow> out;
    out.reserve((size_t)layer->GetFeatureCount());

    OGRFeature* feat = nullptr;
    int global_idx = 0;

    while ((feat = layer->GetNextFeature()) != nullptr) {
        DataRow r;
        r.global_index = global_idx++;

        // attributes
        r.PROVINCIA = has_prov ? get_string_field(feat, defn, spec.provincia_field) : "";
        r.COMUNE    = has_comu ? get_string_field(feat, defn, spec.comune_field) : "";
        r.median_val = get_double_field(feat, defn, spec.value_field, "Value");

        // geometry -> WKT
        OGRGeometry* geom = feat->GetGeometryRef();
        if (geom) {
            char* wkt = nullptr;
            geom->exportToWkt(&wkt);
            if (wkt) {
                r.geometry_wkt = wkt;
                CPLFree(wkt);
            }
        } else {
            r.geometry_wkt.clear(); // valid: no geometry
        }

        out.push_back(std::move(r));
        OGRFeature::DestroyFeature(feat);
    }

    GDALClose(ds);
    return out;
}

std::vector<DataRow> load_dataset(const LoadSpec& spec) {
    if (spec.format == DataFormat::CSV) {
        auto rows = load_csv_rows(spec.path);
        // Keep CSV behavior: geometry empty, sharding by PROVINCIA (your pipeline)
        // Ensure global_index exists (if CSVLoader doesn't set it)
        for (size_t i = 0; i < rows.size(); ++i) rows[i].global_index = (int)i;
        return rows;
    }

    if (spec.format == DataFormat::SHP || spec.format == DataFormat::GPKG) {
        return load_vector_rows_ogr(spec);
    }

    throw std::runtime_error("Unsupported DataFormat");
}
