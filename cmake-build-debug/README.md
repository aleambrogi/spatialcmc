# Consensus Monte Carlo Algorithm for Large Areal Datasets

## Overview
This project implements a scalable, **divide-and-conquer Bayesian nonparametric methodology** for clustering large spatial areal datasets. It addresses the computational limitations of traditional Markov Chain Monte Carlo (MCMC) methods, which often struggle with large-scale spatial data due to their sequential nature.

By leveraging **Consensus Monte Carlo (CMC)** and parallel computing, this framework splits data into non-overlapping shards, performs independent parallel MCMC inference, and aggregates the results into a globally consistent partition using a distance-based consensus mechanism.

## Key Features
* **Scalability:** Reduces computational complexity from $O(n)$ (sequential) to $O(n/S)$ (parallel).
* **Spatial Awareness:** Implements **Spatial Product Partition Models (aPPM)** to preserve geographical cohesion and adjacency structure.
* **Advanced Sampling Engines:**
    * **Neal's Algorithm 2:** For conjugate models (e.g., Gaussian likelihoods).
    * **Neal's Algorithm 8:** For non-conjugate models (e.g., Beta likelihoods).
* **Consensus Mechanism:** Aggregates local results using **Wasserstein distance** and a **Cohesion-Repulsion** likelihood to ensure global consistency.
* **High Performance:** Written entirely in **C++** with a modular design for maximum multi-core CPU utilization.

## Project Structure
The codebase is organized into modular components handling the full Bayesian workflow:

* **`Step0/`**: Data ingestion and spatial structuring. Handles loading of Shapefiles/GPKG and construction of adjacency graphs (KNN or Polygon contiguity).
* **`Step1_Sharding/`**: Logic for partitioning the dataset into non-overlapping shards (e.g., by administrative borders).
* **`MCMC/`**: The parallel Bayesian engine. Contains implementations of Neal's algorithms (2 & 8) and the spatial aPPM priors.
* **`Consensus/`**: Logic for aggregating local results into a global partition using distance-based logic.
* **`utils/`**: Helper functions for matrix operations and spatial calculations.
* **`output/`**: Pipelines for exporting results to CSV and GPKG formats for external analysis.

## Methodological Workflow
1.  **Data Sharding:** The dataset $y$ is split into $S$ subsets based on administrative regions.
2.  **Parallel Inference:** Independent MCMC chains run on each shard to approximate the local posterior $\pi_{s}(\theta|y_{s})$.
3.  **Consensus:** Local partitions are unified into a global clustering $c^*$ by minimizing the **Binder Loss** function against a consensus matrix derived from Wasserstein distances.

## Requirements
To build and run this project, you need:
* **C++ Compiler:** Supporting C++17 or higher.
* **CMake:** Version 3.10+.
* **GDAL/OGR:** For spatial data loading (Shapefiles, GeoPackage).
* **nlohmann/json:** For parsing configuration files.

## Authors
**Francesco Colombo, Mattia Alborghetti, Alessandro Ambrogi, Riccardo Besana, Camilla Carrese, Luca Cerlini** *Tutor: Matteo Gianella*

---
*This framework was validated on the **Healthcare Accessibility Index** across Italy, proving the ability to unify spatial local partitions with global latent structures.*