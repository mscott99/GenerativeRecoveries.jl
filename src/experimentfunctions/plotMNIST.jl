"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::Vector{<:Integer}, images::Vector{<:AbstractArray}; recoveryfn=recoversignal, presigmoid=true, inrange=true, rng=TaskLocalRNG(), plotwidth=600, kwargs...)
    #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
    decoder, truesignals, truesignals_to_plot = setupExperimentDecoderandImages(VAE, images, presigmoid, inrange)

    recoveryerrors = Matrix{Float32}(undef, length(images), length(aimedmeasurementnumbers))
    plotimages = Matrix{typeof(images[1])}(undef, length(images), length(aimedmeasurementnumbers))

    F = fouriermatrix(length(images[1]))
    n = length(images[1])
    @threads for (i, truesignal) in collect(enumerate(truesignals))
        @threads for (j, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            freq = rand(rng, Bernoulli(aimedm / n), n)
            @views sampledF = F[freq, :]
            measurements = sampledF * reshape(truesignal, :)
            recovery = recoveryfn(measurements, sampledF, decoder; kwargs...)
            recoveryerrors[i, j] = norm(recovery .- truesignal)
            plotimages[i, j] = presigmoid ? sigmoid(recovery) : recovery
        end
    end

    f = Figure(resolution=(200 * (length(aimedmeasurementnumbers) + 1), 200 * length(images) + 100), backgroundcolor=:lightgrey)
    Label(f[1, 1], "signal", tellheight=true, tellwidth=false, textsize=20)
    for (i, signalimage) in enumerate(truesignals_to_plot)
        ax = Axis(f[i+1, 1], aspect=1)
        hidedecorations!(ax)
        CairoMakie.heatmap!(ax, 1.0f0 .- signalimage[:, end:-1:1], colormap=:grays)
    end
    for i in 1:size(plotimages)[1], j in 1:size(plotimages)[2]
        ax = Axis(f[i+1, j+1], aspect=1, title="err: $(@sprintf("%.1E", recoveryerrors[i, j]))")
        hidedecorations!(ax)
        heatmap!(ax, 1.0f0 .- plotimages[i, j][:, end:-1:1], colormap=:grays)
    end
    for (i, m) in enumerate(aimedmeasurementnumbers)
        Label(f[1, i+1], "m:$m", tellheight=true, tellwidth=false, textsize=20)
    end
    f
end


@generated function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::Union{Integer,Vector{<:Integer}}, numbers::Union{Integer,Vector{<:Integer}}; typeofdata=:test, rng=TaskLocalRNG(), kwargs...)

    if aimedmeasurementnumbers <: Integer
        measnum = :([aimedmeasurementnumbers])
    elseif aimedmeasurementnumbers <: Vector{<:Integer}
        measnum = :(aimedmeasurementnumbers)
    else
        throw(MethodError(plot_MNISTrecoveries, (VAE, aimedmeasurementnumbers, numbers)))
    end

    return quote
        images = imagesfromnumbers(numbers, typeofdata, rng=rng)
        plot_MNISTrecoveries(VAE, $measnum, images, rng=rng; kwargs...)
    end
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


function plot_MNISTrecoveries(recoveryfns::Vector{<:Function}, VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}; rng=TaskLocalRNG(), typeofdata=:test, kwargs...)
    #TODO incorporate this into the main mrecovery method with the recovery function as parameter.
    images = imagesfromnumbers(numbers, typeofdata, rng=rng)
    plots = []
    for fnpick in recoveryfns
        @time myplot = plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, images; recoveryfn=fnpick, rng=rng, kwargs...)
        push!(plots, myplot)
    end
    plots
end


function plot_MNISTrecoveries(VAE::FullVae, aimedmeasurementnumbers::AbstractArray{<:Integer}, numbers::AbstractArray{<:Integer}, recoveryfunctions::AbstractArray; seed=53, kwargs...)
    plots = []

    for recoveryfn in recoveryfunctions
        rng = Xoshiro(seed)
        #numbersets = [MNISTtestdata.features[:,:, MNISTtestdata.targets.== number] for number in 1:9]
        push!(plots, plot_MNISTrecoveries(VAE, aimedmeasurementnumbers, numbers, recoveryfn=recoveryfn, rng=rng; kwargs...))
    end
    plots
end
