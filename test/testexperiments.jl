@testset "check runexperiments" begin
    using GenerativeRecoveries: plot_recovery_errors_tocompare_frequencysamplingalgorithms, samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically, FullVae, recoversignal, wrap_model_withreshape, VaeEncoder, _setupmodels, _setupMNISTimagesignals, _setupfrequencies, IndexedMatrix, runexperimenttensor
    using GenerativeRecoveries: samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically
    using DataFrames: DataFrame
    using FFTW: plan_dct
    using LinearAlgebra: norm
    using DataFrames: DataFrame

    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    secondmodel = wrap_model_withreshape(model)

    model = _setupmodels(secondmodel)[1]
    images = [2, 3, 5]
    aimedmeasurementnumbers = [22, 33, 45]
    truesignals = _setupMNISTimagesignals(images, secondmodel)
    freqs = _setupfrequencies(aimedmeasurementnumbers, size(truesignals[1]))
    pdct = plan_dct(truesignals[1])
    pdct * truesignals[1]

    img_size = (28, 28)
    numbers = [2, 3, 4]
    aimed_ms = [16, 128]
    #check no error
    plot_recovery_errors_tocompare_frequencysamplingalgorithms(model, numbers, aimed_ms, samplefrequenciesuniformly, samplefrequenciesuniformlyanddeterministically, img_size; max_iter=5)

    #FatFFTPlan(pdct, frequencies[1])
    # mystery error here
    freqs = _setupfrequencies([16, 32, 64], size(truesignals[1]))

    # As = [IndexedMatrix(ParallelMatrix(pdct), freq) for freq in freqs]
    #As = map(x -> FatFFTPlan(pdct, x), frequencies)

    experimentsetup = (truesignals, freqs)
    setupsymbols = (:truesignal, :frequency)
    recoveryfn = recoversignal

    function experimentfn(truesignal, freq, pdct, decoder, recoveryfn; kwargs...) # pass frequencies only
        A = IndexedMatrix(deepcopy(pdct), freq)
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, decoder, max_iter=10; kwargs...)
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        ((:recovered_img => recoveryimg), (:relativeerr => relativeerr))
    end

    array_in = (experimentsetup[i][1] for i in eachindex(experimentsetup))
    decoder = model.decoder
    experiment_result = experimentfn(array_in..., pdct, decoder, recoveryfn, max_iter=10)

    first = true
    df = nothing # initialize symbol for scope purposes

    #pairgenerator = (PairwithArrayIterator(key, enumerate(value)) for (key, value) in experimentsetup)
    setupsymbolsindex = (Symbol(String(key) * "_index") for key in setupsymbols)
    kwargs = (; max_iter=10)
    include_values = true
    args = (pdct, decoder, recoveryfn)


    @test runexperimenttensor(experimentfn, ([freqs[1]],), (:frequency,), images[1], pdct, decoder, recoveryfn) isa DataFrame
    @test runexperimenttensor(experimentfn, experimentsetup, (:truesignal, :frequency), pdct, decoder, recoveryfn; max_iter=10, multithread=false) isa DataFrame
end

@testset "test plotMNIST" begin
    using GenerativeRecoveries: plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder

    @load "../savedmodels/more_incoherentepoch20" model
    model = wrap_model_withreshape(model)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false)
    plot_MNISTrecoveries(model, [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=true)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, presigmoid=false)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, presigmoid=false)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false, presigmoid=false)
    plot_MNISTrecoveries(model, [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false, presigmoid=false)
    #secondmodel = model
    #@load "../savedmodels/bounded_morecoherencematchingepoch20" model
    #model = wrap_model_withreshape(model)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32, 64], [2, 3, 8, 9], max_iter=10, multithread=true)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, presigmoid=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, presigmoid=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=false, inrange=false, presigmoid=false)
    #plot_MNISTrecoveries([model, secondmodel], [16, 32], [2, 3], max_iter=10, multithread=true, inrange=false, presigmoid=false)
    #end



end
# recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
# firstmodel = model