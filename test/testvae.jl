#using BSON: @load

#Base.global_logger(TBLogger("./reusefiles/logs/"))
#Logging.global_logger(Logging.ConsoleLogger())

@info("name", test = 0.1, other = 0.2)
@info other = 0.4

@testset "VAE" begin
    using GenerativeRecoveries

    @load "../savedmodels/bounded_morecoherencematchingepoch20" model
    plot_MNISTrecoveries(model, [16, 32], [1, 2], inrange=false, presigmoid=false)
    recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])
    firstmodel = model

    @load "../savedmodels/incoherentepoch20" model
    secondmodel = model
    plot_models_recovery_errors([firstmodel, secondmodel], ["Bounded", "Incoherent"], [32, 64, 128, 512], inrange=false, presigmoid=false)
end

#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001))