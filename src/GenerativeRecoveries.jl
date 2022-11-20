module GenerativeRecoveries
using BSON
using BSON: @save, @load
using Distributions: Bernoulli
using Base.Threads
using Images, LsqFit, Plots, Printf, DataFrames
using Random, Flux, MLDatasets, FFTW, LinearAlgebra
using Logging, TensorBoardLogger
using CairoMakie: Axis, CairoMakie, Figure, Label, hidedecorations!, heatmap!
using Flux: Chain, params, gradient, update!, Adam
using FFTW: plan_r2r, REDFT00
using Infiltrator: @infiltrate

include("utils.jl")
include("base.jl")
include("VaeModels.jl")
include("experimentfunctions/index.jl")
include("relaxedrecovery.jl")
include("scripts.jl")

function test()
    println("test4")
end


export relaxed_recover, test, addreshape_tomodel
export plot_MNISTrecoveries, compare_models_MNISTrecoveries, recoverythreshold_fromrandomimage
export compare_models_from_thresholds, plot_models_recovery_errors
export FullVae, VaeEncoder, makeMNISTVae, trainVae, trainstdVaeonMNIST, train_incoherentVAE_onMNIST

end # module Generative_Recoveries