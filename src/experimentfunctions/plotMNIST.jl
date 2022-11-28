
function _uniformly_sampled_frequencies(aimedm, img_size; rng=TaskLocalRNG())
    freqs = rand(rng, Bernoulli(aimedm / prod(img_size)), img_size...)
    convert(Array{Bool}, freqs)
end

function _getmodel(model::FullVae, presigmoid=true)
    if !presigmoid #preprocess the models -> add a sigmoid to the decoder. The saved models do not have a sigmoid
        lastlayer = model.decoder.layers[end]
        if typeof(lastlayer) <: Dense && lastlayer.bias == zero(lastlayer.bias)
            newlastlayer = Dense(lastlayer.weight, false, sigmoid)
            newdecoder = Chain(model.decoder.layers[1:end-1]..., newlastlayer)
            VAE = FullVae(model.encoder, newdecoder)
        else
            throw("Unimplemented")
        end
    end
    model
end

"""
Standardise the image inputs
"""
function _getMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::FullVae, datasplit=:test, presigmoid=true, inrange=true; rng=TaskLocalRNG())
    data = MNIST(Float32, datasplit)
    images = []
    for number in numbers
        numberset = data.features[:, :, data.targets.==number]
        push!(images, numberset[:, :, rand(rng, 1:size(numberset)[end])])
    end
    convert(AbstractArray{typeof(images[1])}, images)

    if inrange
        truesignals = fullmodel.(images, 10, rng=rng)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        truesignals = (in -> inversesigmoid.(in)).(images) # double broadcast
    end
    truesignals
end

function _getMNISTimagesignals(numbers::Integer, model::FullVae, datasplit=:test, presigmoid=true, inrange=true; kwargs...)
    _getMNISTimagesignals([numbers], model, datasplit, presigmoid, inrange; kwargs...)
end


function _preprocess_forplot_MNISTsignals(signals::AbstractArray{<:Matrix}, presigmoid)
    presigmoid ? (x -> sigmoid.(x)).(signals) : signals
end



function _getsampledfrequencies(aimed_m::Integer, img_size; rng=TaskLocalRNG())
    rand(rng, Bernoulli(aimed_m / prod(img_size)), img_size...)
end

function _getsampledfrequencies(aimed_m::Integer, p::AbstractArray; kwargs...)
    rand.(Bernoulli.(p))
end

function _plot_tableofrecoveries(plottedtruesignals, recovered_signals::Matrix{<:Matrix{<:AbstractFloat}}, recovery_errors::Matrix{<:AbstractFloat}, array_num_measurements; plotwidth=200)
    numfrequencies = size(recovered_signals, 1)
    numnumbers = size(recovered_signals, 2)
    f = Figure(resolution=(plotwidth * (numnumbers + 1), plotwidth * numfrequencies + plotwidth / 2), backgroundcolor=:lightgrey)
    Label(f[1, 1], "signal", tellheight=true, tellwidth=false, textsize=20)
    for (i, signalimage) in enumerate(plottedtruesignals)
        ax = Axis(f[i+1, 1], aspect=1)
        hidedecorations!(ax)
        heatmap!(ax, 1.0f0 .- signalimage[:, end:-1:1], colormap=:grays)
    end

    for i in 1:numfrequencies, j in 1:numnumbers
        ax = Axis(f[i+1, j+1], aspect=1, title="err: $(@sprintf("%.1E", recovery_errors[i, j]))")
        hidedecorations!(ax)
        heatmap!(ax, 1.0f0 .- recovered_signals[i, j][:, end:-1:1], colormap=:grays)
    end
    for (i, m) in enumerate(array_num_measurements)
        Label(f[1, i+1], "m:$m", tellheight=true, tellwidth=false, textsize=20)
    end
    f
end

"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, numbers; recoveryfn=recoversignal, presigmoid=true, inrange=true, datasplit=:test, kwargs...)
    model = _getmodel(model, presigmoid)
    decoder = model.decoder
    truesignals = _getMNISTimagesignals(numbers, model, datasplit, presigmoid, inrange; kwargs...)

    frequencies = _getsampledfrequencies.(aimedmeasurementnumbers, [size(truesignals[1]),])
    pdct = plan_dct(truesignals[1])

    experimentsetup = [truesignals, frequencies]
    @infiltrate false

    function experimentfn(truesignal, freq, decoder; kwargs...)
        A = fatFFTPlan(pdct, freq)
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, decoder; kwargs...)::Matrix{Float32}
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        (recoveryimg, relativeerr)
    end

    results = runexperimenttensor(experimentfn, experimentsetup, decoder; kwargs...)
    recovered_signals = map(x -> getindex(x, 1), results)
    recovery_errors = map(x -> getindex(x, 2), results)
    plottedtruesignals = _preprocess_forplot_MNISTsignals(truesignals, presigmoid)
    recovered_signals = _preprocess_forplot_MNISTsignals(recovered_signals, presigmoid)
    _plot_tableofrecoveries(plottedtruesignals, recovered_signals, recovery_errors, sum.(frequencies); kwargs...)
end

"Compare many models; this plots the recoveries for each model, keeping the measurements and signal images consistent as much as possible"
function compare_models_MNISTrecoveries(models::Vector{<:FullVae}, aimedmeasurementnumbers, numbers; typeofdata=:test, rng=TaskLocalRNG(), kwargs...)
    returnplots = []
    images = imagesfromnumbers(numbers, typeofdata, rng=rng)
    seed = rand(rng, 1:500)
    for vae in models
        rng = Xoshiro(seed)
        push!(returnplots, plot_MNISTrecoveries(vae, aimedmeasurementnumbers, numbers, rng=rng; kwargs...))
    end
    returnplots
end


# function plot_MNISTrecoveries(recoveryfns::Vector{<:Function}, VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}; rng=TaskLocalRNG(), typeofdata=:test, kwargs...)
#     #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
#     images = imagesfromnumbers(numbers, typeofdata, rng=rng)
#     plots = []
#     for fnpick in recoveryfns
#         @time myplot = plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, images; recoveryfn=fnpick, rng=rng, kwargs...)
#         push!(plots, myplot)
#     end
#     plots
# end


# function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}, recoveryfunctions::AbstractArray; seed=53, kwargs...)
#     plots = []

#     for recoveryfn in recoveryfunctions
#         rng = Xoshiro(seed)
#         #numbersets = [MNISTtestdata.features[:,:, MNISTtestdata.targets.== number] for number in 1:9]
#         push!(plots, plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, numbers, recoveryfn=recoveryfn, rng=rng; kwargs...))
#     end
#     plots
# end
