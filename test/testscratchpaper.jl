# Testing plotMNIST
using GenerativeRecoveries: plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder
using Flux
using BSON: @load
using Revise
@load "savedmodels/more_incoherentepoch20" model
model = wrap_model_withreshape(model)
plot_MNISTrecoveries(model, [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=false)
plot_MNISTrecoveries(model, [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=true)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, presigmoid=false)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, presigmoid=false)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false, presigmoid=false)
plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false, presigmoid=false)
secondmodel = model
@load "savedmodels/bounded_morecoherencematchingepoch20" model
model = wrap_model_withreshape(model)
plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=true)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, presigmoid=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, presigmoid=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false, presigmoid=false)
plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false, presigmoid=false)


# Test runexperiment tensor
using Test
using Flux
using BSON: @load
using GenerativeRecoveries: FullVae, recoversignal, wrap_model_withreshape, VaeEncoder, _getmodel, _getMNISTimagesignals, _getsampledfrequencies, ParallelMatrix, IndexedMatrix, runexperimenttensor
using FFTW: plan_dct
using LinearAlgebra: norm

@load "savedmodels/bounded_morecoherencematchingepoch20" model
secondmodel = wrap_model_withreshape(model)

@load "savedmodels/more_incoherentepoch20" model
model = wrap_model_withreshape(model)
@test secondmodel isa FullVae

model = _getmodel(secondmodel)
images = [2, 3, 5]
aimedmeasurementnumbers = [22, 33, 45]
truesignals = _getMNISTimagesignals(images, secondmodel)
freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(truesignals[1]))
pdct = plan_dct(truesignals[1])
pdct * truesignals[1]
#FatFFTPlan(pdct, frequencies[1])
# mystery error here
freqs = _getsampledfrequencies([16, 32, 64], size(truesignals[1]))

# As = [IndexedMatrix(ParallelMatrix(pdct), freq) for freq in freqs]
#As = map(x -> FatFFTPlan(pdct, x), frequencies)


experimentsetup = (truesignals, freqs)

recoveryfn = recoversignal

function experimentfn(truesignal, freq, pdct, decoder, recoveryfn; kwargs...) # pass frequencies only
    A = IndexedMatrix(pdct, freq)
    A = ParallelMatrix(A)
    measurements = A * truesignal
    recoveryimg = recoveryfn(measurements, A, decoder, max_iter=10; kwargs...)
    relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
    (recoveryimg, relativeerr)
end

array_in = (experimentsetup[i][1] for i in 1:length(experimentsetup))

decoder = model.decoder
# As[1] * truesignals[1]

typeof(truesignals[1])
size(truesignals[1])

experiment_result = experimentfn(array_in..., pdct, decoder, recoveryfn, max_iter=10)

@test runexperimenttensor(experimentfn, ([freqs[1]],), images[1], pdct, decoder, recoveryfn) isa Vector
@test runexperimenttensor(experimentfn, experimentsetup, pdct, decoder, recoveryfn, max_iter=10, multithread=true) isa Matrix
# struct ParallelFFTPlan{T<:AbstractFFTs.Plan}
#     pdct::AbstractArray{T}
# end

# ParallelFFTPlan(pdct::AbstractFFTs.Plan) = ParallelFFTPlan([deepcopy(pdct) for i in 1:nthreads()])

# *(a::ParallelFFTPlan, x::AbstractArray) = a.pdct[threadid()] * x

# struct FatFFTPlan{T<:Union{AbstractFFTs.Plan,ParallelFFTPlan},F<:AbstractArray{Bool}}
#     p::T
#     freqs::F
# end

# *(A::FatFFTPlan, x::AbstractArray) = (A.p*x)[A.freqs]

# struct IndexedMatrix{T,L}
#     A::T
#     indices::L
# end

# import Base: *

# *(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

# using Base.Threads

# indices = reshape(collect(CartesianIndices(length.(experimentsetup))), :)

# decoder = secondmodel.decoder

# @threads for index in indices
#     results[index] = experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., decoder, recoveryfn, max_iter=3000)
# end

# results


# using Colors: Gray
# using GenerativeRecoveries: logrange
# @time plot_MNISTrecoveries(wrapped_model, logrange(16, 783, 6), [1, 2, 3, 4, 5, 6], max_iter=3000, multithread=false)

# @test plot_MNISTrecoveries([wrapped_model, secondmodel], [2, 4], [3, 5], max_iter=10) isa AbstractArray{<:Figure,1}

# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], relaxed_recover, optimlayers=[2], max_iter=10) isa Figure
# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], [relaxed_recover, recoversignal], optimlayers=[2], max_iter=10) isa AbstractArray{<:Figure}
# using Base.Threads

# indices = reshape(collect(CartesianIndices(length.(experimentsetup))), :)

# decoder = secondmodel.decoder

# @threads for index in indices
#     results[index] = experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., decoder, recoveryfn, max_iter=3000)
# end

# results


# using Colors: Gray
# using GenerativeRecoveries: logrange
# @time plot_MNISTrecoveries(wrapped_model, logrange(16, 783, 6), [1, 2, 3, 4, 5, 6], max_iter=3000, multithread=false)

# @test plot_MNISTrecoveries([wrapped_model, secondmodel], [2, 4], [3, 5], max_iter=10) isa AbstractArray{<:Figure,1}

# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], relaxed_recover, optimlayers=[2], max_iter=10) isa Figure
# @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], [relaxed_recover, recoversignal], optimlayers=[2], max_iter=10) isa AbstractArray{<:Figure}




