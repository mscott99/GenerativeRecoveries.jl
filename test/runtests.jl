using MLDatasets
using Flux
using Flux: @epochs, train!, params, DataLoader
using CUDA
using Test: @testset, @test

#include("../src/GenerativeRecoveries.jl")
using GenerativeRecoveries

include("./testvae.jl")