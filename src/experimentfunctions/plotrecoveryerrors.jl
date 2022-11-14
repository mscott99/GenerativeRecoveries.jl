"""
Make a scatter plot of recovery errors for random images for different numbers of measurements.
"""
function plot_models_recovery_errors(models::Vector{<:FullVae}, modellabels::Vector{<:AbstractString}, aimedmeasurementnumbers::AbstractArray;
    presigmoid=true, inrange=true, typeofdata=:test, savefile="reusefiles/experiment_data/ansdata.BSON", kwargs...)

    if !presigmoid #preprocess the models
        for model in models
            model.decoder = sigmoid ∘ model.decoder
        end
    end

    dataset = MNIST(Float32, typeofdata).features

    returnplot = plot()
    returndata = Dict()

    returnplot = plot()
    for (label, model) in zip(modellabels, models)
        recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))
        true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))

        @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
            img = dataset[:, :, rand(1:size(dataset)[3])]
            truesignal, _ = _preprocess_MNIST_truesignal(img, model, presigmoid, inrange)

            true_m, F = sampleFourierwithoutreplacement(aimedm, getlayerdims(model.decoder)[end], true)
            measurements = F * truesignal
            recovery = recoversignal(measurements, F, model.decoder; kwargs...)

            true_ms[i] = true_m
            recoveryerrors[i] = norm(recovery .- truesignal)
        end

        returndata[label] = (true_ms, recoveryerrors)
        returnplot = scatter!(true_ms, recoveryerrors, yaxis=:log, label=label)
    end

    if !isnothing(savefile)
        @save savefile returndata inrange presigmoid aimedmeasurementnumbers
    end
    returnplot
end
