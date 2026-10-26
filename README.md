# SpatialCMC

**Bayesian clustering in C++: Dirichlet process mixtures, adjacency-aware sampling and consensus clustering.**

Research code exploring how mixture models can incorporate neighbourhood structure and how sampled partitions or component parameters can be summarised. The repository contains the original modelling work plus runnable synthetic examples and a portable build.

## What is implemented

- A univariate Gaussian Dirichlet process mixture with a Normal–Inverse-Gamma base prior and Gibbs sampling.
- An adjacency-aware extension (`aPPM_MixtureModel`).
- A distance-based cohesion–repulsion sampler for consensus experiments.
- CSV loading, province-based sharding and asynchronous shard-level sampling.

The examples demonstrate methods on synthetic data; they are not a benchmark or a validated production system.

## Build and run

Requires **CMake 3.20+** and a **C++20 compiler**. No external C++ libraries are needed.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
./build/spatialcmc dp
./build/spatialcmc spatial
./build/spatialcmc consensus
```

On Visual Studio generators, the executable is normally `build/Release/spatialcmc.exe`. With MinGW, add `-G "MinGW Makefiles"` to the configure command. Executables print sampled assignments and component summaries to the terminal.

## Repository guide

| Location | Contents |
|---|---|
| `MCMC/` | Prior/model interface, Gaussian mixture samplers and asynchronous sampling |
| `Consensus/` | Cohesion–repulsion sampler |
| `Step0/` | Original CSV reader |
| `Step1_Sharding/` | Grouping observations into shards |
| `main.cpp` | Three synthetic examples and an experimental CSV pipeline function |
| `tests/` | Small sampler checks for draw sizes, labels and finite positive variances |

## Research notes and limitations

The old CSV-to-meta-clustering experiment is preserved in `main.cpp` but is not exposed by the command line: its original `dataset_median.csv` is retained for research context, shard adjacency assumes grouped input order, and partition comparisons across shards require a shared observation set. In that experiment the k-nearest-neighbour graph is constructed from values, rather than geographic coordinates. It must not be interpreted as a validated spatial pipeline.

The simple CSV reader assumes unquoted delimiter-free cells and is intended for controlled research inputs. Sampler seeds are not consistently configurable across all components; repeated runs need not be identical. A successful smoke check establishes basic execution, not MCMC convergence or statistical accuracy.

## Credits

Original authorship and collaborators remain recorded in the repository's Git history and source headers. Portfolio maintenance preserves the modelling implementation; it adds build instructions, examples, checks and repository housekeeping.

No new license is imposed by this cleanup. Third-party inputs retain their original terms.

## Verified build

On 26 October 2026, the project configured and built with CMake 3.24.1 and GCC 15.2.0 (MinGW, Windows). The sampler smoke test passed, and all three synthetic examples completed successfully. This checks execution rather than convergence or predictive performance.
