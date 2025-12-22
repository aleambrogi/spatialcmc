if ("package:spatialcmc" %in% search()) {
  detach("package:spatialcmc", unload = TRUE, character.only = TRUE)
}


dlls <- getLoadedDLLs()
if ("spatialcmc" %in% names(dlls)) {
  try(dyn.unload(dlls[["spatialcmc"]]$path), silent = TRUE)
}

setwd("C:/Users/samsung/Desktop/spatialcmc-rcpp")

unlink("src/RcppExports.cpp")
unlink("R/RcppExports.R")


Rcpp::compileAttributes()
devtools::install(build = TRUE, dependencies = FALSE)

library(spatialcmc)

ls("package:spatialcmc")
getDLLRegisteredRoutines("spatialcmc")


result <- run_spatialcmc_full_cpp(
  "C:/Users/samsung/Desktop/spatialcmc/dataset_median.csv",
  "beta",
  1.0,
  0.5,
  100,
  200
)

str(result)  # Per vedere cosa ti ritorna

