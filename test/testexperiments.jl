using BSON: @load
using Flux
using Infiltrator: @infiltrate
#Base.global_logger(TBLogger("./reusefiles/logs/"))
#Logging.global_logger(Logging.ConsoleLogger())
push!(LOAD_PATH, pwd())
using Revise
using GenerativeRecoveries

function test_addreshape_tomodel()
    @load "savedmodels/incoherentepoch20" model
    new_model = addReshapeToModel(model)
    test_image = MNIST(:test)[1].features
    plot(plot(Gray.(test_image)'), plot(Gray.(new_model(test_image, 10))'))
end




@load "savedmodels/bounded_morecoherencematchingepoch20" model
model.decoder

new_decoder = Chain(model.decoder..., x -> reshape(x, 28, 28))
using Plots: plot
using Colors
using MLDatasets: MNIST

test_image = MNIST(:test)[1].features

@show new_decoder
test_image =
    code = randn(16)
plot(Gray.(new_decoder(code)))



plot_MNISTrecoveries(model, [16, 32], [1, 2])
recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
firstmodel = model

@load "savedmodels/incoherentepoch20" model
secondmodel = model
plot_models_recovery_errors([firstmodel, secondmodel], ["Bounded", "Incoherent"], [32, 64, 128, 512], inrange=false, presigmoid=false)

struct dctMatrix::AbstractMatrix{Float64}
    n::Int
end

import Base: getindex
getindex(A::dctMatrix, i::Int, j::Int) = cos((i - 1) * (j - 1) * π / A.n)
using MLDatasets
testimage = MNIST(:test)[1].features
using Plots, Colors
plot(Gray.(dct(testimage')))

using FFTW: dct
dct(testimage)

using Distributions: Bernoulli
aimedm = 10
truesignals = [testimage]
freq = rand(Bernoulli(aimedm / length(truesignals[1])), size(truesignals[1])...)
sum(freq)
using FFTW: dct
using Zygote: gradient, Params

othersignal = rand(28, 28)
measurements = dct(othersignal)[freq]
gradient(() -> (dct(truesignals[1]), truesignals[1]))

gradient(() -> sum(x -> x^2, dct(truesignals[1])), Params(truesignals[1]))


signal = randn(8)
gradient(() -> sum(x -> x^2, dct(signal)), Params(signal))

using LinearAlgebra: norm



gradient(() -> sum(dct(signal)), Params(signal))
using AbstractFFTs: pifft
f3(x) = real(norm(pifft.scale .* (pifft.p * fftshift(pfft * x))))

f3(signal)
using FFTW: plan_r2r, REDFT00
pdct = plan_r2r(truesignals[1], REDFT00)
gradient(() -> sum(((pdct*truesignals[1])[freq] .- measurements) .^ 2), Params(truesignals[1]))




#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001) )