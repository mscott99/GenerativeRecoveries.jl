include("src/GenerativeRecoveries.jl")
using .GenerativeRecoveries

#dataset = FileDataset("~/.julia/DataDeps/CELEBA/img_align_celeba/")

using BSON: @load
using Flux

@load "savedmodels/bounded_morecoherencematchingepoch20" model

plot_MNISTrecoveries(model, 64, [1, 2, 4, 5], inrange=false, presigmoid=false)

recoverythreshold_fromrandomimage(model, [32, 64, 128, 512])

@load "reusefiles/savedmodels/bounded_morecoherencematchingepoch20" model
boundedmodel = model


f = compare_models_MNISTrecoveries([incoherentmodel, boundedmodel], 128, 8, inrange=false, presigmoid=true)
f[1]
f[2]
@info f

firstplot
secondplot



@time plot_MNISTrecoveries(model, logrange(32, 784, 5), [2, 3, 4, 5, 6, 7, 8, 9], inrange=false, presigmoid=false)

#TODO: Compare with rng



f = Figure()

@generated function tester(x; arg=2)
    return :(x * x)
end
tester(2)
throw(MethodError(plot_MNISTrecoveries))