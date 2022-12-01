
@testset "test runExperimentTensor" begin
    using GenerativeRecoveries: ParallelMatrix
    #label to remember order
    experimentsetuplabels = ["frequencies", "images"]
    # get the sampled frequencies
    aimed_ms = [10, 20]

    freqs = _getsampledfrequencies(aimed_ms, (28, 28))

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
        A = ParallelMatrix(pdct, freq)
        measurements = A * truesignal
        recovery = recoversignal(measurements, A, decoder; kwargs...)
        recovery
    end

    result = runexperimenttensor(experimentfn, [freqs, images], pdct)
    @test result isa Array

    @test runexperimenttensor(experimentfn, [], freqs[1], images[1], pdct) isa Array{<:Matrix,0}
    @test runexperimenttensor(experimentfn, [[freqs[1]]], images[1], pdct) isa Array{<:Matrix,1}


end