#using Infiltrator
#using CairoMakie
#using CairoMakie: heatmap, plot
#using Flux
#using Infiltrator: @infiltrate
#using Revise
#using Colors
#using CairoMakie: heatmap

# Test loading the model
@testset "test plot_MNISTrecoveries function" begin
    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    wrapped_model = wrap_model_withreshape(model)
    @test wrapped_model isa FullVae

    test_image = MNIST(:test)[rand(1:200)].features
    @test typeof(test_image) == typeof(wrapped_model(test_image, 2))

    @test plot_MNISTrecoveries(wrapped_model, [2, 3], [3, 4], max_iter=10) isa Figure

    @test plot_MNISTrecoveries(wrapped_model, 2, 3, max_iter=10) isa Figure

    @load "../savedmodels/more_incoherentepoch20" model
    secondmodel = wrap_model_withreshape(model)
    @test secondmodel isa FullVae

    @test plot_MNISTrecoveries([wrapped_model, secondmodel], [2, 4], [3, 5], max_iter=10) isa AbstractArray{<:Figure,1}

    @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], relaxed_recover, optimlayers=[2], max_iter=10) isa Figure
    @test plot_MNISTrecoveries(wrapped_model, [2, 4], [3, 5], [relaxed_recover, recoversignal], optimlayers=[2], max_iter=10) isa AbstractArray{<:Figure}
end
# recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
# firstmodel = model