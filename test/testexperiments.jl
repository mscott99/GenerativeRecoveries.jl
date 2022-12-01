#using Infiltrator
#using CairoMakie
#using CairoMakie: heatmap, plot
#using Flux
#using Infiltrator: @infiltrate
#using Revise
#using Colors
#using CairoMakie: heatmap
using Flux
using GenerativeRecoveries
using GenerativeRecoveries: runexperimenttensor, wrap_model_withreshape
using GenerativeRecoveries: recoversignal, addreshape_tomodel, FullVae, logrange, runexperimenttensor, wrap_model_withreshape, _getsampledfrequencies
using BSON: @load
using MLDatasets
using Flux
using Test
using Distributions: Bernoulli
using FFTW
using FFTW: plan_dct
using Base.Threads
# Test loading the model
#@testset "test plot_MNISTrecoveries function" begin
#@test wrapped_model isa FullVae

#test_image = MNIST(:test)[rand(1:200)].features
#@test typeof(test_image) == typeof(wrapped_model(test_image, 2))

#@test plot_MNISTrecoveries(wrapped_model, [2, 3], [3, 4], max_iter=10) isa Figure

#@test plot_MNISTrecoveries(wrapped_model, 2, 3, max_iter=10) isa Figure
@testset "check runexperiments" begin
    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    secondmodel = wrap_model_withreshape(model)

    using BSON: @load
    @load "../savedmodels/more_incoherentepoch20" model
    model = wrap_model_withreshape(model)
    @test secondmodel isa FullVae

    #plot_MNISTrecoveries(model, 80, 7)

    using GenerativeRecoveries: _getmodel, _getMNISTimagesignals, _getsampledfrequencies, ParallelMatrix, IndexedMatrix

    model = _getmodel(secondmodel)[1]
    images = [2, 3, 5]
    aimedmeasurementnumbers = [22, 33, 45]
    truesignals = _getMNISTimagesignals(images, secondmodel)
    freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(truesignals[1]))

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

    pdct = plan_dct(truesignals[1])
    pdct * truesignals[1]
    #FatFFTPlan(pdct, frequencies[1])
    # mystery error here
    freqs = _getsampledfrequencies([16, 32, 64], size(truesignals[1]))

    # As = [IndexedMatrix(ParallelMatrix(pdct), freq) for freq in freqs]
    #As = map(x -> FatFFTPlan(pdct, x), frequencies)


    experimentsetup = (truesignals, freqs)

    using GenerativeRecoveries: recoversignal
    using LinearAlgebra: norm
    recoveryfn = recoversignal

    # struct IndexedMatrix{T,L}
    #     A::T
    #     indices::L
    # end

    # import Base: *

    # *(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

    using GenerativeRecoveries: IndexedMatrix, ParallelMatrix

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

    @time runexperimenttensor(experimentfn, experimentsetup, pdct, decoder, recoveryfn, max_iter=10, multithread=true)

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
end


@testset "test plotMNIST" begin
    using GenerativeRecoveries: plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder
    using Flux
    using BSON: @load
    @load "../savedmodels/more_incoherentepoch20" model
    model = wrap_model_withreshape(model)
    @time plot_MNISTrecoveries(model, [16, 32, 64], [2, 3, 8, 9], max_iter=100, multithread=true)
    @time plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=100, multithread=true, inrange=false)
    @time plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=100, multithread=true, inrange=false, presigmoid=false)
end



#end
# recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
# firstmodel = model