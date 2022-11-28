using MLDatasets
using Flux
using Flux: @epochs, train!, params, DataLoader
using Test: @testset, @test
using Distributions: Bernoulli
using FFTW: plan_dct

#include("../src/GenerativeRecoveries.jl")
using GenerativeRecoveries
using BSON: @load

include("./testexperiments.jl")
include("./testutils.jl")