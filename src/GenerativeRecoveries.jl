module GenerativeRecoveries
using CairoMakie
using AlgebraOfGraphics
using BSON
using MLDatasets: MNIST, FunctionalFileDataset, FunctionalSubDataset
using BSON: @save, @load
using Distributions: Bernoulli
using Base.Threads
using Images, LsqFit, Printf
using Random, Flux, MLDatasets, FFTW, LinearAlgebra, Statistics
using Logging, TensorBoardLogger
using ProgressLogging: @progress, Progress
using CairoMakie: Axis, CairoMakie, Figure, Label, hidedecorations!, heatmap!
using Flux: Chain, gradient, update!, Adam, logitbinarycrossentropy, pullback, DataLoader, params
using FFTW: plan_dct, plan_fft
using StatsBase: sample
using DataFrames
using DataFrames: allcombinations
using Infiltrator: @infiltrate
using SpecialFunctions: gamma

include("VaeModels.jl")
include("measurementsampling.jl")
include("recoveryalgorithms.jl")
include("preprocess.jl")
include("plot.jl")
include("utils.jl")
include("experimentfunctions/index.jl")


export relaxed_recover, runexperimenttensor, wrap_model_withreshape
export plot_MNISTrecoveries, compare_models_MNISTrecoveries, recoverythreshold_fromrandomimage
export compare_models_from_thresholds, plot_models_recovery_errors
export FullVae, VaeEncoder, makeMNISTVae, trainVae, trainstdVaeonMNIST, train_incoherentVAE_onMNIST

end # module Generative_Recoveries