using GenerativeRecoveries: _uniformly_sampled_frequencies, fatFFTPlan, recoversignal

@testset "test function runExperimentTensor" begin
    #label to remember order
    experimentsetuplabels = ["frequencies", "images"]
    # get the sampled frequencies
    aimed_ms = [10, 20]
    freqs = [_uniformly_sampled_frequencies(aimed_m, (28, 28)) for aimed_m in aimed_ms]

    # Get the signals
    images = [MNIST(:test)[1].features, MNIST(:test)[2].features]

    # Get the model. Wrap it.
    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    function wrapmodel_withreshape(model::FullVae)
        new_decoder = Chain(model.decoder..., x -> reshape(x, 28, 28))
        wrapped_model = FullVae(model.encoder, new_decoder)
    end
    model = wrapmodel_withreshape(model)
    @test model isa FullVae

    decoder = model.decoder

    experimentsetup = [freqs, images]

    pdct = plan_dct(images[1])

    function experimentfn(freq, truesignal, pdct; kwargs...)
        A = fatFFTPlan(pdct, freq)
        measurements = A * truesignal
        recovery = recoversignal(measurements, A, decoder; kwargs...)
        recovery
    end

    result = runexperimenttensor(experimentfn, [freqs, images], pdct)
    @test result isa Array
    #using CairoMakie: heatmap
    #heatmap(result[1, 2])
end