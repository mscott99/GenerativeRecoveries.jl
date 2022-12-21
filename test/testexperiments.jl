@testset "test plotMNIST" begin
    using GenerativeRecoveries: plot_MNISTrecoveryerrors, plot_MNISTrecoveries, wrap_model_withreshape, FullVae, VaeEncoder, samplefrequenciesuniformlyanddeterministically
    using GenerativeRecoveries: samplesmallestfrequencies, samplefrequenciesuniformly
    @load "../savedmodels/more_incoherentepoch20" model
    model = wrap_model_withreshape(model)

    plot_MNISTrecoveryerrors(model, [samplesmallestfrequencies, samplefrequenciesuniformlyanddeterministically], [2, 5], [1, 2], max_iter=3)
    plot_MNISTrecoveryerrors(model, [samplesmallestfrequencies, samplefrequenciesuniformly], [2, 5], [1, 2], max_iter=3, presigmoid=false, dct=true, inrange=false)

    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3)
    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3, presigmoid=false, inrange=false, dct=true)
    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3, inrange=false, dct=true)
    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3, inrange=false, presigmoid=false, samplingfn=samplefrequenciesuniformlyanddeterministically)
    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3, inrange=false, presigmoid=false, samplingfn=samplesmallestfrequencies)
    plot_MNISTrecoveries(model, [2, 4], [2, 3], max_iter=3, inrange=false, presigmoid=false, dct=true)

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