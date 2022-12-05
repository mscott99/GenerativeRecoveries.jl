
"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, images, recoveryfn::Function=recoversignal; presigmoid=true, inrange=true, datasplit=:test, multithread=false, kwargs...)

    function experimentfn(truesignal, frequency, model, pdct, recoveryfn; multithread=true, kwargs...)
        if multithread
            pdct = deepcopy(pdct)
        end
        A = IndexedMatrix(pdct, frequency)
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, model; kwargs...)
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        ((:recovered_signal => recoveryimg), (:relative_error => relativeerr))
    end

    model = _getmodel(model; presigmoid)
    decoder = model.decoder
    truesignals = _getMNISTimagesignals(images, model; datasplit, presigmoid, inrange, kwargs...)
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(truesignals[1]))
    pdct = plan_dct(truesignals[1])

    setuplabels = (:truesignal, :frequency)
    experimentsetup = (truesignals, freqs)
    fixedsetup = (decoder, pdct, recoveryfn)
    resultdataframe = runexperimenttensor(experimentfn, experimentsetup, setuplabels, fixedsetup...; kwargs...)

    _plot_tables_ofrecoveries(resultdataframe; presigmoid) # in progress
end

#function _plot_tableofrecoveries(df::DataFrame; presigmoid, kwargs...)
#if presigmoid
#transform!(df, :recovered_signal => (x-> sigmoid.(x)))
#end
#for plotbykey in [:recoveryfn, :model]
#if(plotbykey in names(df))
#grouped_df =groupby(df, plotbykey)
#for subframe in grouped_df

#end
#end
#end
##groupedframe = groupby(df, )
#end
function _plot_tables_ofrecoveries(df::DataFrame; kwargs...)
    if :model in names(df)
        grouped_index = :model
    elseif :recoveryfn in names(df)
        grouped_index = :recoveryfn
    else
        return _plot_tableofrecoveries(df; kwargs...)
    end

    grouped_df = groupby(df, grouped_index)
    figures = []
    for gdf in grouped_df
        push!(figures, _plot_tableofrecoveries(df; kwargs...))
    end
    return figures
end

using DataFrames: transform!, select, select!, transform
"Plot table of recoveries by frequency and recovery signals"
function _plot_tableofrecoveries(df::AbstractDataFrame; presigmoid=true, plotwidth=200, kwargs...)
    if presigmoid
        df = transform(df, :recovered_signal => sigmoid => :recovered_signal)
        transform!(df, :truesignal => sigmoid => :truesignal)
    end
    numfrequencies = length(unique(df[!, :frequency_index]))
    numnumbers = length(unique(df[!, :truesignal_index]))
    f = Figure(resolution=(plotwidth * (numfrequencies + 1), plotwidth * numnumbers + plotwidth / 2), backgroundcolor=:lightgrey)
    Label(f[1, 1], "signal", tellheight=true, tellwidth=false, textsize=20)

    signalimages = unique(df[:, [:truesignal, :truesignal_index]])
    for signalrow in eachrow(signalimages)
        ax = Axis(f[signalrow[:truesignal_index]+1, 1], aspect=1)
        hidedecorations!(ax)
        heatmap!(ax, 1.0f0 .- signalrow[:truesignal][:, end:-1:1], colormap=:grays)
    end

    for row in eachrow(df)
        df = unique(df) # in case there are other experiment dimensions
        ax = Axis(f[row[:truesignal_index]+1, row[:frequency_index]+1], aspect=1, title="err: $(@sprintf("%.1E", row[:relative_error]))")
        hidedecorations!(ax)
        heatmap!(ax, 1.0f0 .- row[:recovered_signal][:, end:-1:1], colormap=:grays)
    end

    # compute the true frequency numbers
    frequencies_num_measurements = unique(select(df, :frequency_index, :frequency))
    select!(frequencies_num_measurements, :frequency_index, :frequency => (x -> sum.(x)) => :m)

    for frequency_instance in eachrow(frequencies_num_measurements)
        Label(f[1, frequency_instance[:frequency_index]+1], "m:$(frequency_instance[:m])", tellheight=true, tellwidth=false, textsize=20)
    end
    f
end

#function _plot_tableofrecoveries(plottedtruesignals, recovered_signals::Matrix{<:Matrix{<:AbstractFloat}}, recovery_errors::Matrix{<:AbstractFloat}, array_num_measurements; plotwidth=200, kwargs...)
#numfrequencies = size(recovered_signals, 1)
#numnumbers = size(recovered_signals, 2)
#f = Figure(resolution=(plotwidth * (numnumbers + 1), plotwidth * numfrequencies + plotwidth / 2), backgroundcolor=:lightgrey)
#Label(f[1, 1], "signal", tellheight=true, tellwidth=false, textsize=20)
#for (i, signalimage) in enumerate(plottedtruesignals)
#ax = Axis(f[i+1, 1], aspect=1)
#hidedecorations!(ax)
#heatmap!(ax, 1.0f0 .- signalimage[:, end:-1:1], colormap=:grays)
#end
#for i in 1:numfrequencies, j in 1:numnumbers
#ax = Axis(f[i+1, j+1], aspect=1, title="err: $(@sprintf("%.1E", recovery_errors[i, j]))")
#hidedecorations!(ax)
#heatmap!(ax, 1.0f0 .- recovered_signals[i, j][:, end:-1:1], colormap=:grays)
#end
#for (i, m) in enumerate(array_num_measurements)
#Label(f[1, i+1], "m:$m", tellheight=true, tellwidth=false, textsize=20)
#end
#f
#end

#function plot_MNISTrecoveries(models::Vector{<:FullVae}, aimedmeasurementnumbers, images; datasplit=:test, kwargs...)
## other method for array of models, because the models impact everything
#images = _getMNISTimagesignals(images, datasplit; kwargs...) #to standardise images and frequencies
#freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(images[1]); kwargs...)
#[plot_MNISTrecoveries(model, freqs, images; datasplit=datasplit, kwargs...) for model in models]
#end

#function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, images, recovery_functions::AbstractArray{<:Function}; datasplit=:test, kwargs...)
#images = _getMNISTimagesignals(images, datasplit; kwargs...) #to standardise images and frequencies
#freqs = _getsampledfrequencies(aimedmeasurementnumbers, size(images[1]); kwargs...)
#[plot_MNISTrecoveries(model, freqs, images, recoveryfn; datasplit=datasplit, kwargs...) for recoveryfn in recovery_functions]
#end



"""Pre-process models for recovery experiments"""
function _getmodel(model::FullVae, presigmoid=true; kwargs...)
    if presigmoid
        return model
    else
        lastlayer = model.decoder.layers[end]
        if typeof(lastlayer) <: Dense && lastlayer.bias == zero(lastlayer.bias)
            newlastlayer = Dense(lastlayer.weight, false, sigmoid)
            newdecoder = Chain(model.decoder.layers[1:end-1]..., newlastlayer)
            return FullVae(model.encoder, newdecoder)
        elseif lastlayer isa Function # Assume it's a reshape
            secondtolastlayer = model.decoder.layers[end-1]
            secondtolastlayer = Dense(secondtolastlayer.weight, false, sigmoid)
            newdecoder = Chain(model.decoder.layers[1:end-2]..., secondtolastlayer, lastlayer)
            return FullVae(model.encoder, newdecoder)
        else
            throw("Unimplemented")
        end
    end
end


"""Pre-process models for recovery experiments"""
function _getmodels(models::AbstractArray{<:FullVae}, presigmoid=true; kwargs...)
    results = []
    for model in models
        if presigmoid
            push!(results, model)
        else
            lastlayer = model.decoder.layers[end]
            if typeof(lastlayer) <: Dense && lastlayer.bias == zero(lastlayer.bias)
                newlastlayer = Dense(lastlayer.weight, false, sigmoid)
                newdecoder = Chain(model.decoder.layers[1:end-1]..., newlastlayer)
                push!(results, FullVae(model.encoder, newdecoder))
            elseif lastlayer isa Function # Assume it's a reshape
                secondtolastlayer = model.decoder.layers[end-1]
                secondtolastlayer = Dense(secondtolastlayer.weight, false, sigmoid)
                newdecoder = Chain(model.decoder.layers[1:end-2]..., secondtolastlayer, lastlayer)
                push!(results, FullVae(model.encoder, newdecoder))
            else
                throw("Unimplemented")
            end
        end
    end
    return convert(Array{typeof(results[1])}, results)
end

function _getmodels(model::FullVae; kwargs...)
    _getmodels([model]; kwargs...)
end



# function _getMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::AbstractArray{<:FullVae}; datasplit=:test, presigmoid=true, inrange=true, rng=TaskLocalRNG(), kwargs...)



"""
Standardise the image inputs
"""
function _getMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::FullVae; datasplit=:test, presigmoid=true, inrange=true, rng=TaskLocalRNG(), kwargs...)
    images = _getMNISTimagesignals(numbers; datasplit)
    if inrange
        return fullmodel.(images, 10, rng=rng)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        return (in -> inversesigmoid.(in)).(images) # double broadcast
    else
        return images
    end
end

function _getMNISTimagesignals(numbers::Vector{<:Integer}; datasplit=:test, rng=TaskLocalRNG(), kwargs...)
    data = MNIST(Float32, datasplit)
    [data.features[:, :, data.targets.==number][:, :, rand(rng, 1:size(data.features[:, :, data.targets.==number])[end])] for number in numbers]
end

function _getMNISTimagesignals(number::Integer, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    _getMNISTimagesignals([number], model; datasplit, presigmoid, inrange, kwargs...)
end

function _getMNISTimagesignals(images::AbstractArray{<:Matrix}, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    if inrange
        return model.(images, 10)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        return (in -> inversesigmoid.(in)).(images) # double broadcast
    else
        return images
    end
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