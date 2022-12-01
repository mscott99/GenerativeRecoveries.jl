@testset "check runexperiments" begin
    using GenerativeRecoveries: FullVae, recoversignal, wrap_model_withreshape, VaeEncoder, _getmodel, _getMNISTimagesignals, _getsampledfrequencies, ParallelMatrix, IndexedMatrix, runexperimenttensor
    using FFTW: plan_dct
    using LinearAlgebra: norm

    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    secondmodel = wrap_model_withreshape(model)

    @load "../savedmodels/more_incoherentepoch20" model
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
end

@testset "test plotMNIST" begin
    using GenerativeRecoveries: plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder

    @load "../savedmodels/more_incoherentepoch20" model
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
    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    model = wrap_model_withreshape(model)
    plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=true)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, presigmoid=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, presigmoid=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false, presigmoid=false)
    plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false, presigmoid=false)
end



#end
# recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
# firstmodel = model