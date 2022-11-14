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

include("base.jl")
include("VaeModels.jl")
include("experimentfunctions/index.jl")
include("relaxedrecovery.jl")

export relaxed_recover
export plot_MNISTrecoveries, compare_models_MNISTrecoveries, recoverythreshold_fromrandomimage
export compare_models_from_thresholds, plot_models_recovery_errors
export FullVae, VaeEncoder, makeVae, trainVae, trainstdVaeonMNIST, train_incoherentVAE_onMNIST

end # module Generative_Recoveries