using BSON: @load

using Infiltrator
using CairoMakie
#using CairoMakie: heatmap, plot
using Flux
#using Infiltrator: @infiltrate
using Revise
using GenerativeRecoveries
using GenerativeRecoveries: addreshape_tomodel, FullVae, logrange, runexperimenttensor
#using Colors
#using CairoMakie: heatmap
using MLDatasets: MNIST
using Test

# Test loading the model
@testset "loading a model" begin
    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    new_decoder = Chain(model.decoder..., x -> reshape(x, 28, 28))
    wrapped_model = FullVae(model.encoder, new_decoder)
    @test wrapped_model isa FullVae

    test_image = MNIST(:test)[rand(1:200)].features
    @test typeof(test_image) == typeof(wrapped_model(test_image, 2))

    @test plot_MNISTrecoveries(wrapped_model, [2, 3], [3, 4]) isa Figure

    @test plot_MNISTrecoveries(wrapped_model, 2, 3) isa Figure
end

# recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
# firstmodel = model

# @load "savedmodels/incoherentepoch20" model
# secondmodel = model
# plot_models_recovery_errors([firstmodel, secondmodel], ["Bounded", "Incoherent"], [32, 64, 128, 512], inrange=false, presigmoid=false)

# struct dctMatrix::AbstractMatrix{Float64}
#     n::Int
# end

# import Base: getindex
# getindex(A::dctMatrix, i::Int, j::Int) = cos((i - 1) * (j - 1) * π / A.n)
# using MLDatasets
# testimage = MNIST(:test)[1].features
# using Plots, Colors
# plot(Gray.(dct(testimage')))

# using FFTW: dct
# dct(testimage)

# using Distributions: Bernoulli
# aimedm = 10
# truesignals = [testimage]
# freq = rand(Bernoulli(aimedm / length(truesignals[1])), size(truesignals[1])...)
# sum(freq)
# using FFTW: dct
# using Zygote: gradient, Params

# othersignal = rand(28, 28)
# measurements = dct(othersignal)[freq]
# gradient(() -> (dct(truesignals[1]), truesignals[1]))

# gradient(() -> sum(x -> x^2, dct(truesignals[1])), Params(truesignals[1]))


# signal = randn(8)
# gradient(() -> sum(x -> x^2, dct(signal)), Params(signal))

# using LinearAlgebra: norm



# gradient(() -> sum(dct(signal)), Params(signal))
# using AbstractFFTs: pifft
# f3(x) = real(norm(pifft.scale .* (pifft.p * fftshift(pfft * x))))

# f3(signal)
# using FFTW: plan_r2r, REDFT00
# pdct = plan_r2r(truesignals[1], REDFT00)
# gradient(() -> sum(((pdct*truesignals[1])[freq] .- measurements) .^ 2), Params(truesignals[1]))




#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001) )