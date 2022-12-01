# using MLDatasets
# using Flux
# using Flux: @epochs, train!, params, DataLoader
# using Test
# using Distributions: Bernoulli
# using FFTW: plan_dct
# using BSON: @load
# using CairoMakie: Figure
# using GenerativeRecoveries
# using GenerativeRecoveries: recoversignal, addreshape_tomodel, FullVae, logrange, runexperimenttensor, wrap_model_withreshape, _getsampledfrequencies

using Test
using Flux
using BSON: @load

include("./testexperiments.jl")
include("./testutils.jl")