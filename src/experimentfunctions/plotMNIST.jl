
"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, images, recoveryfn::Function=recoversignal; presigmoid=true, inrange=true, datasplit=:test, multithread=false, kwargs...)
    model = _getmodel(model, presigmoid)[1]
    decoder = model.decoder
    truesignals = _getMNISTimagesignals(images, model, datasplit, presigmoid, inrange; kwargs...)
    freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(truesignals[1]))

    pdct = plan_dct(truesignals[1])
    # As = [IndexedMatrix(ParallelMatrix(pdct), freq) for freq in freqs]
    #if multithread
    #    As = [ParallelMatrix(A) for A in As]
    #end

    experimentsetup = (truesignals, freqs)

    function experimentfn(truesignal, freq, pdct, decoder, recoveryfn; multithread=multithread, kwargs...)
        A = IndexedMatrix(pdct, freq)
        if multithread
            A = ParallelMatrix(A)
        end
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, decoder; kwargs...)
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        (recoveryimg, relativeerr)
    end

    results = runexperimenttensor(experimentfn, experimentsetup, pdct, decoder, recoveryfn, multithread=multithread; kwargs...)
    recovered_signals = map(x -> getindex(x, 1), results)
    recovery_errors = map(x -> getindex(x, 2), results)
    plottedtruesignals = _preprocess_forplot_MNISTsignals(truesignals, presigmoid)
    recovered_signals = _preprocess_forplot_MNISTsignals(recovered_signals, presigmoid)
    _plot_tableofrecoveries(plottedtruesignals, recovered_signals, recovery_errors, sum.(freqs); kwargs...)
end

function _getmodel(models::Array{<:FullVae}, presigmoid=true; kwargs...)
    if !presigmoid
        for (i, model) in models
            lastlayer = model.decoder.layers[end]
            if typeof(lastlayer) <: Dense && lastlayer.bias == zero(lastlayer.bias)
                newlastlayer = Dense(lastlayer.weight, false, sigmoid)
                newdecoder = Chain(model.decoder.layers[1:end-1]..., newlastlayer)
                models[i] = FullVae(model.encoder, newdecoder)
            else
                throw("Unimplemented")
            end
        end
    end
    models
end

function _getmodel(model::FullVae, presigmoid=true; kwargs...)
    _getmodel([model], presigmoid)
end

"""
Standardise the image inputs
"""
function _getMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::FullVae, datasplit=:test, presigmoid=true, inrange=true; rng=TaskLocalRNG(), kwargs...)
    images = _getMNISTimagesignals(numbers, datasplit)
    if inrange
        truesignals = fullmodel.(images, 10, rng=rng)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        truesignals = (in -> inversesigmoid.(in)).(images) # double broadcast
    end
    truesignals
end

function _getMNISTimagesignals(numbers::Vector{<:Integer}, datasplit=:test; rng=TaskLocalRNG(), kwargs...)
    data = MNIST(Float32, datasplit)
    [data.features[:, :, data.targets.==number][:, :, rand(rng, 1:size(data.features[:, :, data.targets.==number])[end])] for number in numbers]
end

function _getMNISTimagesignals(numbers::Integer, model::FullVae, datasplit=:test, presigmoid=true, inrange=true; kwargs...)
    _getMNISTimagesignals([numbers], model, datasplit, presigmoid, inrange; kwargs...)
end

function _getMNISTimagesignals(images::AbstractArray{<:Matrix}, model::FullVae, datasplit=:test, presigmoid=true, inrange=true; kwargs...)
    if inrange
        truesignals = model.(images, 10)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        truesignals = (in -> inversesigmoid.(in)).(images) # double broadcast
    end
    truesignals
end

function _getsampledfrequencies(pre_sampled_frequencies::Vector{<:AbstractArray{<:Bool}}, img_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    pre_sampled_frequencies
end

function _getsampledfrequencies(pre_sampled_frequencies::AbstractArray{<:Bool}, img_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    [pre_sampled_frequencies]
end

function _getsampledfrequencies(aimed_ms::Vector{<:Integer}, img_size::Tuple{<:Integer,<:Integer}; rng=TaskLocalRNG(), kwargs...)
    [rand(rng, Bernoulli(aimed_m / prod(img_size)), img_size...) for aimed_m in aimed_ms]
end

function _getsampledfrequencies(aimed_m::Integer, img_size::Tuple{<:Integer,<:Integer}; rng=TaskLocalRNG(), kwargs...)
    _getsampledfrequencies([aimed_m], img_size; rng=rng)
end

function _getsampledfrequencies(p::AbstractArray, ; kwargs...)
    [rand.(Bernoulli.(p))]
end

function _preprocess_forplot_MNISTsignals(signals::AbstractArray{<:Matrix}, presigmoid; kwargs...)
    presigmoid ? (x -> sigmoid.(x)).(signals) : signals
end

function _plot_tableofrecoveries(plottedtruesignals, recovered_signals::Matrix{<:Matrix{<:AbstractFloat}}, recovery_errors::Matrix{<:AbstractFloat}, array_num_measurements; plotwidth=200, kwargs...)
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

function _plot_tableofrecoveries(plottedtruesignals, recovered_signals::Array{<:Matrix{<:AbstractFloat},3}, recovery_errors::Array{<:AbstractFloat}, array_num_measurements; plotwidth=200, kwargs...)
    [_plot_tableofrecoveries(plottedtruesignals, recovered_signals[:, :, i], recovery_errors[:, :, i], array_num_measurements; plotwidth=plotwidth) for i in 1:size(recovered_signals, 3)]
end
"Returns an array of plots, one for each model"
function plot_MNISTrecoveries(models::Vector{<:FullVae}, aimedmeasurementnumbers, images; datasplit=:test, kwargs...)
    # other method for array of models, because the models impact everything
    images = _getMNISTimagesignals(images, datasplit; kwargs...) #to standardise images and frequencies
    freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(images[1]); kwargs...)
    [plot_MNISTrecoveries(model, freqs, images; datasplit=datasplit, kwargs...) for model in models]
end

function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, images, recovery_functions::AbstractArray{<:Function}; datasplit=:test, kwargs...)
    images = _getMNISTimagesignals(images, datasplit; kwargs...) #to standardise images and frequencies
    freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(images[1]); kwargs...)
    [plot_MNISTrecoveries(model, freqs, images, recoveryfn; datasplit=datasplit, kwargs...) for recoveryfn in recovery_functions]
end