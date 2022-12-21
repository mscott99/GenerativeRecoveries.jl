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
    Label(f[1, 1], "signal", tellheight=true, tellwidth=false, fontsize=20)

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
        Label(f[1, frequency_instance[:frequency_index]+1], "m:$(frequency_instance[:m])", tellheight=true, tellwidth=false, fontsize=20)
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