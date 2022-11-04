
#Base.global_logger(TBLogger("./reusefiles/logs/"))
#Logging.global_logger(Logging.ConsoleLogger())

@info("name", test = 0.1, other = 0.2)
@info other = 0.4

@testset "VAE" begin
    @test 1 == 1
end

#include("./trainloops.jl")
#trainlognsave(loss,)

#does not work
#train!(vaeloss(vaemodel, 0.5, 0.5) ,params(vaemodel), loader, Flux.Optimise.ADAM(0.001))