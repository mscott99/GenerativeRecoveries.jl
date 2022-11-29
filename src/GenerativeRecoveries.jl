module GenerativeRecoveries
using BSON
using BSON: @save, @load
using Distributions: Bernoulli
using Base.Threads
using Images, LsqFit, Plots, Printf
using Random, Flux, MLDatasets, FFTW, LinearAlgebra
using Logging, TensorBoardLogger
using CairoMakie: Axis, CairoMakie, Figure, Label, hidedecorations!, heatmap!
using Flux: Chain, params, gradient, update!, Adam
using FFTW: plan_dct

include("VaeModels.jl")
include("utils.jl")
include("base.jl")
include("experimentfunctions/index.jl")
include("relaxedrecovery.jl")
include("scripts.jl")


export relaxed_recover, addreshape_tomodel, runexperimenttensor
export plot_MNISTrecoveries, compare_models_MNISTrecoveries, recoverythreshold_fromrandomimage
export compare_models_from_thresholds, plot_models_recovery_errors
export FullVae, VaeEncoder, makeMNISTVae, trainVae, trainstdVaeonMNIST, train_incoherentVAE_onMNIST

end # module Generative_Recoveries