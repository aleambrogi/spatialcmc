#include "BayesianPipelineBaseline.h"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include "ogrsf_frmts.h"
#include "cpl_error.h" // Required
#include "cpl_conv.h" // Required for CPLSetConfigOption

// ------------------------------------------------------------
// Population Method: Fills the object with pipeline results
// ------------------------------------------------------------
void BayesianPipelineBaseline::populate_from_pipeline(
    std::vector<std::vector<int>> clusterings,           // Passed by value (Copy or Move happened at call site)
    std::vector<std::map<int, ComponentParameter>> parameters,
    std::vector<ComponentParameter> phi,
    std::vector<int> meta_labels,
    SamplerType sampler,
    double alpha_dp_, double lambda_appm_,
    int burn_in_, int samples_,
    double alpha_meta_, double delta_in_, double lambda_in_,
    double delta_out_, double theta_out_,
    int best_meta_iteration_,
    bool has_conjugate_prior_,
    double mu0_, double kappa0_,
    double alpha0_, double beta0_
) {
    // INTERNAL MOVE: We move the local 'value' into the member variable.
    // This is instant.
    all_clusterings = std::move(clusterings);
    all_parameters  = std::move(parameters);
    phi_data        = std::move(phi);
    best_meta_labels= std::move(meta_labels);

    sampler_type = sampler;
    alpha_dp = alpha_dp_;
    lambda_appm = lambda_appm_;
    burn_in = burn_in_;
    samples = samples_;
    alpha_meta = alpha_meta_;
    delta_in = delta_in_;
    lambda_in = lambda_in_;
    delta_out = delta_out_;
    theta_out = theta_out_;
    best_meta_iteration = best_meta_iteration_;
    has_conjugate_prior = has_conjugate_prior_;
    if (has_conjugate_prior) {
        mu0 = mu0_; kappa0 = kappa0_; alpha0 = alpha0_; beta0 = beta0_;
    }

    compute_global_posteriors_from_meta();
    map_local_to_global();
}

// ------------------------------------------------------------
// Mapping Logic: Local Clusters -> Global Meta Clusters
// ------------------------------------------------------------
void BayesianPipelineBaseline::map_local_to_global() {
    if(all_parameters.empty()) return;

    // OPTIMIZATION: Use Flat Vector Lookup instead of std::map
    // 1. Build lookup table: table[shard][local_id] = global_id
    std::vector<std::vector<int>> lookup_table(all_parameters.size());
    int flat_index = 0;

    for (size_t shard_idx = 0; shard_idx < all_parameters.size(); ++shard_idx) {
        int max_local_id = 0;
        if (!all_parameters[shard_idx].empty()) {
            max_local_id = all_parameters[shard_idx].rbegin()->first;
        }
        lookup_table[shard_idx].assign(max_local_id + 1, -1);

        for (const auto& kv : all_parameters[shard_idx]) {
            if (flat_index >= best_meta_labels.size()) break;
            int local_label = kv.first;
            int global_label = best_meta_labels[flat_index];

            if(local_label >= 0) lookup_table[shard_idx][local_label] = global_label;
            flat_index++;
        }
    }

    // 2. Fast Assign
    global_assignments.clear();
    global_assignments.resize(all_clusterings.size());

    // Use OpenMP if you have it, otherwise normal loop is fine
    #pragma omp parallel for schedule(dynamic)
    for (size_t shard_idx = 0; shard_idx < all_clusterings.size(); ++shard_idx) {
        const auto& shard_clusters = all_clusterings[shard_idx];
        std::vector<int> current_shard_globals(shard_clusters.size());

        for (size_t i = 0; i < shard_clusters.size(); ++i) {
            int local_c = shard_clusters[i];
            int global_c = -1;
            if (local_c >= 0 && local_c < lookup_table[shard_idx].size()) {
                global_c = lookup_table[shard_idx][local_c];
            }
            current_shard_globals[i] = global_c;
        }
        global_assignments[shard_idx] = std::move(current_shard_globals);
    }
    std::cout << "Mapped global assignments for " << all_clusterings.size() << " shards.\n";
}

std::map<int, int>
BayesianPipelineBaseline::build_global_index_to_cluster_map(
    const std::vector<std::vector<DataRow>>& shards) const
{
    std::map<int, int> global_map;

    for (size_t s = 0; s < shards.size(); ++s) {
        for (size_t i = 0; i < shards[s].size(); ++i) {

            int global_idx = shards[s][i].global_index;
            int global_c   = global_assignments[s][i];

            global_map[global_idx] = global_c;
        }
    }

    return global_map;
}

// ------------------------------------------------------------
// Posterior Computation
// ------------------------------------------------------------
void BayesianPipelineBaseline::compute_global_posteriors_from_meta() {
    global_posteriors.clear();
    for (size_t i = 0; i < phi_data.size(); ++i) {
        int meta = best_meta_labels[i];
        global_posteriors[meta] = phi_data[i]; // Left unchanged
    }
}

// ------------------------------------------------------------
// Output & Saving Methods
// ------------------------------------------------------------
void BayesianPipelineBaseline::save_meta_mcmc_point_trace(
    const std::string& filename,
    const std::vector<std::vector<int>>& meta_samples,
    const std::vector<DataRow>& original_data_sorted
) {
    if (meta_samples.empty()) {
        std::cerr << "No meta-samples provided.\n";
        return;
    }

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    // 1. Build a Map: (ShardID, LocalClusterID) -> MetaItemIndex
    // The CohesionRepulsionDP receives parameters in a specific flattened order.
    // We must reconstruct that exact order to know which column in 'meta_samples'
    // corresponds to which local cluster.

    std::map<std::pair<int, int>, int> local_to_meta_idx;
    int flat_idx = 0;

    // Iterate exactly as we did when building phi_data in SpatialCMCInterface
    for (size_t s = 0; s < all_parameters.size(); ++s) {
        // Map is sorted by key (LocalClusterID), ensuring deterministic order
        for (const auto& kv : all_parameters[s]) {
            int local_id = kv.first;
            local_to_meta_idx[{ (int)s, local_id }] = flat_idx;
            flat_idx++;
        }
    }

    // Check consistency
    if (meta_samples[0].size() != (size_t)flat_idx) {
        std::cerr << "Warning: Mismatch between meta-sample columns ("
                  << meta_samples[0].size() << ") and total local clusters (" << flat_idx << ").\n";
    }

    // 2. Write Header
    file << "MetaIteration";

    // Calculate total points
    int total_points = 0;
    for (const auto& shard_c : all_clusterings) total_points += shard_c.size();

    if (!original_data_sorted.empty() && original_data_sorted.size() == total_points) {
        for (const auto& row : original_data_sorted) file << "," << row.COMUNE;
    } else {
        for (int i = 0; i < total_points; ++i) file << ",Pt_" << i;
    }
    file << "\n";

    // 3. Write Trace Rows
    std::cout << "Saving meta-point trace (" << meta_samples.size() << " iterations) to " << filename << "...\n";

    for (size_t t = 0; t < meta_samples.size(); ++t) {
        file << t; // Iteration

        const auto& current_global_config = meta_samples[t];

        // Iterate over every point in the dataset (Shard 0 -> N)
        for (size_t s = 0; s < all_clusterings.size(); ++s) {
            const auto& shard_assignments = all_clusterings[s]; // Fixed local assignment

            for (int local_c : shard_assignments) {

                int global_c = -1;

                // 1. Find which "Meta Item" this local cluster corresponds to
                auto key = std::make_pair((int)s, local_c);
                if (local_to_meta_idx.count(key)) {
                    int meta_idx = local_to_meta_idx.at(key);

                    // 2. Look up the Global ID for that item at this iteration
                    if (meta_idx < current_global_config.size()) {
                        global_c = current_global_config[meta_idx];
                    }
                }

                file << "," << global_c;
            }
        }
        file << "\n";
    }

    file.close();
}

void BayesianPipelineBaseline::save_dataset_with_global_clusters(
    const std::string& filename,
    const std::vector<DataRow>& original_data,
    const std::vector<std::vector<DataRow>>& shards
) const
{
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open output CSV: " + filename);
    }

    // Build map global_index -> cluster
    auto global_map = build_global_index_to_cluster_map(shards);

    // Header
    file << "GlobalIndex,PROVINCIA,COMUNE,Value,GlobalClusterID\n";

    for (const auto& row : original_data) {

        int global_idx = row.global_index;
        int cluster_id = -1;

        if (global_map.count(global_idx)) {
            cluster_id = global_map.at(global_idx);
        }

        file << global_idx << ","
             << "\"" << row.PROVINCIA << "\","
             << "\"" << row.COMUNE << "\","
             << row.median_val << ","
             << cluster_id << "\n";
    }

    file.close();

    std::cout << "Saved dataset with global clusters to "
              << filename << "\n";
}

void BayesianPipelineBaseline::print_summary() const {
    std::cout << "\n================== BAYESIAN PIPELINE SUMMARY ==================\n";

    std::cout << "Sampler type: "
              << (sampler_type == SamplerType::Neal2
                      ? "Neal2 (conjugate)"
                      : "Neal8 (non-conjugate / Beta)")
              << "\n";

    if (has_conjugate_prior) {
        std::cout << "Prior (conjugate NIG): mu0=" << mu0
                  << " kappa0=" << kappa0
                  << " alpha0=" << alpha0
                  << " beta0=" << beta0 << "\n";
    } else {
        std::cout << "Prior: non-conjugate / not available\n";
    }

    std::cout << "DP hyperparameters: alpha=" << alpha_dp
              << " lambda=" << lambda_appm << "\n";

    std::cout << "MCMC: burn-in=" << burn_in
              << " samples=" << samples << "\n";

    std::cout << "Meta-clustering: alpha=" << alpha_meta
              << " delta_in=" << delta_in
              << " lambda_in=" << lambda_in
              << " delta_out=" << delta_out
              << " theta_out=" << theta_out << "\n";

    std::cout << "Best meta-clustering iteration (Binder loss): "
              << best_meta_iteration << "\n";

    std::cout << "Number of chains: " << all_parameters.size() << "\n";
    std::cout << "Number of components: " << phi_data.size() << "\n";
    std::cout << "Best meta-labels size: " << best_meta_labels.size() << "\n";

    std::cout << "\nGlobal posteriors:\n";
    for (const auto& kv : global_posteriors) {
        std::cout << "Cluster " << kv.first << " | params=[";
        const auto& pars = kv.second.parameters;
        for (size_t i = 0; i < pars.size(); ++i) {
            std::cout << pars[i];
            if (i + 1 < pars.size()) std::cout << ", ";
        }
        std::cout << "]\n";
    }


    std::cout << "================================================================\n";
}


void BayesianPipelineBaseline::save_global_clustering_csv(const std::string& filename, const std::vector<DataRow>& original_data) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << "\n";
        return;
    }

    // Header
    file << "GlobalIndex,ShardID,LocalIndexInShard,LocalClusterID,GlobalClusterID";
    // If we have original data, add extra info
    if (!original_data.empty()) {
        file << ",PROVINCIA,COMUNE,Value";
    }
    file << "\n";

    // We assume that original_data is ordered exactly like the sharding.
    // NOTE: Since sharding reorders data in 'shards' vectors, we must be careful.
    // The safest approach is to save the global index if available.
    // However, 'all_clusterings' follows the structure of shards.

    int global_counter_check = 0;

    for (size_t s = 0; s < global_assignments.size(); ++s) {
        for (size_t i = 0; i < global_assignments[s].size(); ++i) {

            int local_c = all_clusterings[s][i];
            int global_c = global_assignments[s][i];

            // Write basic info
            // Note: we don't easily have the original GlobalIndex here unless we pass the Shard object.
            // We will write a sequential ID based on export order or use original data if passed correctly.

            file << global_counter_check << "," << s << "," << i << "," << local_c << "," << global_c;

            if (!original_data.empty() && global_counter_check < original_data.size()) {
                // WARNING: This assumes 'original_data' passed here is ALREADY sorted by Shard
                // If original_data is the raw vector read from CSV, the order does not match!
                // For now we only print if the user passes sorted data, otherwise leave empty.
                // Future implementation: pass the Sharding object or sorted data.
                // Print row data
                const auto& row = original_data[global_counter_check];
                file << ",\"" << row.PROVINCIA << "\",\"" << row.COMUNE << "\"," << row.median_val;
            }

            file << "\n";
            global_counter_check++;
        }
    }

    std::cout << "Saved global clustering results to " << filename << "\n";
    file.close();
}

void BayesianPipelineBaseline::save_global_clustering_gpkg(
    const std::string& filename,
    const std::vector<DataRow>& sorted_data,
    const std::string& layer_name
) const {

    //CPLSetConfigOption("GDAL_DATA", "C:\\Program Files\\GDAL\\gdal-data");

    CPLPushErrorHandler(CPLQuietErrorHandler);
    GDALAllRegister();

    const char *pszDriverName = "GPKG";
    GDALDriver *poDriver = GetGDALDriverManager()->GetDriverByName(pszDriverName);
    if (!poDriver) {
        std::cerr << "GPKG driver not available.\n";
        return;
    }

    // Create the datasource (the file)
    GDALDataset *poDS = poDriver->Create(filename.c_str(), 0, 0, 0, GDT_Unknown, nullptr);
    if (!poDS) {
        std::cerr << "Creation of output file failed: " << filename << "\n";
        return;
    }

    // Create the layer
    // Note: We don't have the SpatialReference (CRS) here easily unless stored in DataRow.
    // The output will be valid geometry but "Unknown CRS" in QGIS.
    OGRLayer *poLayer = poDS->CreateLayer(layer_name.c_str(), nullptr, wkbUnknown, nullptr);
    if (!poLayer) {
        std::cerr << "Layer creation failed.\n";
        GDALClose(poDS);
        return;
    }

    // Define Fields (Schema)
    OGRFieldDefn fProv("PROVINCIA", OFTString);
    poLayer->CreateField(&fProv);

    OGRFieldDefn fCom("COMUNE", OFTString);
    poLayer->CreateField(&fCom);

    OGRFieldDefn fVal("Value", OFTReal);
    poLayer->CreateField(&fVal);

    OGRFieldDefn fGlobal("GlobalIndex", OFTInteger);
    poLayer->CreateField(&fGlobal);

    OGRFieldDefn fCluster("GlobalClusterID", OFTInteger);
    poLayer->CreateField(&fCluster);

    // Add Shard/Local ID for debugging if needed
    OGRFieldDefn fShard("ShardID", OFTInteger);
    poLayer->CreateField(&fShard);

    // Write Data
    // We assume 'sorted_data' matches the order of 'global_assignments' (Shard 0->N)
    size_t global_counter = 0;

    for (size_t s = 0; s < global_assignments.size(); ++s) {
        for (size_t i = 0; i < global_assignments[s].size(); ++i) {

            if (global_counter >= sorted_data.size()) break;

            const auto& row = sorted_data[global_counter];
            int cluster_id = global_assignments[s][i];

            OGRFeature *poFeature = OGRFeature::CreateFeature(poLayer->GetLayerDefn());

            // 1. Reconstruct Geometry from WKT
            if (!row.geometry_wkt.empty()) {
                char* wkt = const_cast<char*>(row.geometry_wkt.c_str());
                OGRGeometry *poGeom = nullptr;
                OGRGeometryFactory::createFromWkt(&wkt, nullptr, &poGeom);
                if (poGeom) {
                    poFeature->SetGeometryDirectly(poGeom); // Feature takes ownership
                }
            }

            // 2. Set Attributes
            poFeature->SetField("PROVINCIA", row.PROVINCIA.c_str());
            poFeature->SetField("COMUNE", row.COMUNE.c_str());
            poFeature->SetField("Value", row.median_val);
            poFeature->SetField("GlobalIndex", row.global_index);
            poFeature->SetField("GlobalClusterID", cluster_id);
            poFeature->SetField("ShardID", (int)s);

            // 3. Commit
            if (poLayer->CreateFeature(poFeature) != OGRERR_NONE) {
                std::cerr << "Failed to create feature " << row.global_index << "\n";
            }

            OGRFeature::DestroyFeature(poFeature);
            global_counter++;
        }
    }

    std::cout << "Saved global clustering to GPKG: " << filename << "\n";
    GDALClose(poDS);
}