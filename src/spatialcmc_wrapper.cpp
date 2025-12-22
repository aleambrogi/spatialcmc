#include <Rcpp.h>
#include "Step0/CSVLoader.h"
#include "Step1_Sharding/Sharding.h"
#include "MCMC/ParallelDPMixture.h"
#include "MCMC/IModelPrior.h"
#include "MCMC/BetaNonConjugateModel.h"
#include "Consensus/CohesionRepulsionDP.h"
#include "BayesianPipelineBaseline.h"

using namespace Rcpp;

// Dummy function per creare adjacency locale (puoi sostituire con la tua reale)
std::vector<std::vector<int>> extract_local_adjacency(
    const std::vector<DataRow>& shard,
    const std::vector<std::vector<int>>& global_adj
){
  std::vector<std::vector<int>> local(shard.size());
  for (size_t i = 0; i < shard.size(); ++i)
    local[i] = {}; // vuoto per ora
  return local;
}

// [[Rcpp::export]]
List run_spatialcmc_full_cpp(
    std::string csv_path,
    std::string model_type,
    double alpha,
    double lambda,
    int burnin,
    int samples
) {
  // 1️⃣ Carica CSV
  CSVReader reader(csv_path);
  std::vector<DataRow> data = reader.read();
  if (data.empty()) stop("CSV is empty");
  
  Rcpp::Rcout << "Loaded " << data.size() << " rows.\n";
  
  // 2️⃣ Sharding
  for (size_t i = 0; i < data.size(); ++i)
    data[i].global_index = i;
  
  Step1_Sharding::Sharding sharder(data);
  auto shards = sharder.generateShards("PROVINCIA");
  Rcpp::Rcout << "Generated " << shards.size() << " shards.\n";
  
  // 3️⃣ Prior
  IModelPrior* prior = nullptr;
  if (model_type == "normal") {
    prior = new NormalConjugateModel(0.0, 1.0, 2.0, 2.0);
  } else if (model_type == "beta") {
    prior = new BetaNonConjugateModel(2,1,2,1,0.25,0.25,2);
  } else {
    stop("Unknown model type");
  }
  
  // 4️⃣ Adjacency locale dummy
  std::vector<std::vector<std::vector<int>>> adjacency_per_shard;
  for (const auto& shard : shards)
    adjacency_per_shard.push_back(extract_local_adjacency(shard, {}));
  
  // 5️⃣ Parallel DP Mixture
  ParallelDPMixture pdp(alpha, shards, prior, adjacency_per_shard, lambda);
  pdp.run_parallel_mcmc(burnin, samples);
  
  // 6️⃣ Recupero risultati
  const auto& all_samples_c   = pdp.get_best_clustering_per_chain();
  const auto& all_samples_phi = pdp.get_best_parameters_per_chain();
  int M = all_samples_c.size();
  
  // 7️⃣ Meta-clustering su phi
  std::vector<ComponentParameter> phi_data;
  for (int ch = 0; ch < M; ++ch)
    for (const auto& kv : all_samples_phi[ch])
      phi_data.push_back(kv.second);
  
  int Q = phi_data.size();
  std::vector<std::vector<double>> D_phi(Q, std::vector<double>(Q, 0.0));
  
  // Calcola distanza (dummy: differenza assoluta su primo parametro)
  for (int i = 0; i < Q; ++i)
    for (int j = i + 1; j < Q; ++j) {
      D_phi[i][j] = D_phi[j][i] = std::abs(phi_data[i].parameters[0] - phi_data[j].parameters[0]);
    }
    
    CohesionRepulsionDP phiDP(5.0, D_phi, 0.5, 10.0, 2.0, 10.0, 999);
  phiDP.initialize_random(6);
  phiDP.run_mcmc(1, 100);
  phiDP.save_meta_cluster_assignments_csv(phi_data, "meta_cluster_assignments.csv");
  
  int best_phi_idx = phiDP.best_sample_binder();
  const auto& final_phi_labels = phiDP.get_samples_labels()[best_phi_idx];
  
  // 8️⃣ Pipeline finale e salvataggio globale
  BayesianPipelineBaseline pipeline;
  double alpha_meta = 5.0, delta_in = 0.5, lambda_in = 10.0, delta_out = 2.0, theta_out = 10.0;
  double mu0 = 0.0, kappa0 = 0.1, alpha0 = 2.0, beta0 = 0.01;
  
  pipeline.populate_from_pipeline(
    all_samples_c, all_samples_phi, phi_data, final_phi_labels,
    SamplerType::Neal2,
    alpha, lambda,
    burnin, samples,
    alpha_meta, delta_in, lambda_in, delta_out, theta_out,
    best_phi_idx,
    true,  // prior coniugata
    mu0, kappa0, alpha0, beta0
  );
  
  pipeline.print_summary();
  
  // Ordina i dati per export
  std::vector<DataRow> sorted_rows_for_export;
  for(const auto& shard : shards)
    for(const auto& row : shard)
      sorted_rows_for_export.push_back(row);
  
  pipeline.save_global_clustering_csv("FINAL_GLOBAL_CLUSTERING.csv", sorted_rows_for_export);
  
  delete prior;
  
  List out;
  out["n_shards"] = shards.size();
  out["alpha"]    = alpha;
  out["lambda"]   = lambda;
  out["Q_phi"]    = Q;
  out["K_meta"]   = *std::max_element(final_phi_labels.begin(), final_phi_labels.end()) + 1;
  return out;
}

// Generato da Rcpp::compileAttributes
RcppExport SEXP _spatialcmc_run_spatialcmc_full_cpp(SEXP csv_pathSEXP, SEXP model_typeSEXP,
                                                    SEXP alphaSEXP, SEXP lambdaSEXP,
                                                    SEXP burninSEXP, SEXP samplesSEXP) {
  BEGIN_RCPP
  Rcpp::RObject rcpp_result_gen;
  Rcpp::RNGScope rcpp_rngScope_gen;
  Rcpp::traits::input_parameter<std::string>::type csv_path(csv_pathSEXP);
  Rcpp::traits::input_parameter<std::string>::type model_type(model_typeSEXP);
  Rcpp::traits::input_parameter<double>::type alpha(alphaSEXP);
  Rcpp::traits::input_parameter<double>::type lambda(lambdaSEXP);
  Rcpp::traits::input_parameter<int>::type burnin(burninSEXP);
  Rcpp::traits::input_parameter<int>::type samples(samplesSEXP);
  rcpp_result_gen = Rcpp::wrap(run_spatialcmc_full_cpp(csv_path, model_type, alpha, lambda, burnin, samples));
  return rcpp_result_gen;
  END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
  {"_spatialcmc_run_spatialcmc_full_cpp", (DL_FUNC) &_spatialcmc_run_spatialcmc_full_cpp, 6},
  {NULL, NULL, 0}
};

RcppExport void R_init_spatialcmc(DllInfo *dll) {
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
