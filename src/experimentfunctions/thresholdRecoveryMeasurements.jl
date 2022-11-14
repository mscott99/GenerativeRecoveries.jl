
"""Fit a sigmoid to data with log in y only"""
function threshold_through_fit(xdata, ydata; sigmoid_x_scale=2.5f0)
    ylogdata = log.(ydata)
    @. curve(x, p) = p[1] * sigmoid((x - p[2]) / (-sigmoid_x_scale)) + p[3]
    p0 = [3.0f0, 1.0f2, 1.5f0]
    fit = curve_fit(curve, xdata, ylogdata, p0)
    scatter(xdata, ydata, yaxis=:log) #although we do not fit for log(x) we still plot x in log scale for clarity
    (coef(fit)[2], fit, plot!(x -> exp(curve(x, coef(fit)))))
end

"""Scatter plot recovery errors for a single image, fit a sigmoid in the log-log scale, return the recovery threshold from the fit"""
function recoverythreshold_fromrandomimage(VAE, aimedmeasurementnumbers; img=nothing, presigmoid=true, inrange=true, typeofdata=:test, savefile="../experiment_data/ansdata.BSON", kwargs...)

    if isnothing(img)
        dataset = MNIST(Float32, typeofdata).features
        img = dataset[:, :, rand(1:size(dataset)[3])]
    end

    # pick image at random
    decoder = VAE.decoder
    if !presigmoid #preprocess the models
        VAE = sigmoid ∘ VAE
        decoder = sigmoid ∘ decoder
    end

    truesignal, _ = _preprocess_MNIST_truesignal(img, VAE, presigmoid, inrange)

    true_ms = Vector{Float32}(undef, length(aimedmeasurementnumbers))
    recoveryerrors = Vector{Float32}(undef, length(aimedmeasurementnumbers))

    @threads for (i, aimedm) in collect(enumerate(aimedmeasurementnumbers))
        true_m, F = sampleFourierwithoutreplacement(aimedm, getlayerdims(decoder)[end], true)
        measurements = F * reshape(truesignal, :)
        recovery = recoversignal(measurements, F, decoder; kwargs...)
        true_ms[i] = true_m
        recoveryerrors[i] = norm(recovery - truesignal)
    end

    threshold, fit, returnplot = threshold_through_fit(true_ms, recoveryerrors)

    datapoints = hcat(true_ms, recoveryerrors)

    if !isnothing(savefile)
        @save savefile true_ms recoveryerrors threshold truesignal inrange presigmoid aimedmeasurementnumbers VAE fit
    end

    (threshold=threshold, fitplot=returnplot, fitdata=datapoints, fitobject=fit) #threshold, and things to check if threshold is accurate
end

"""Compare models through the recovery threshold of a small number of images"""
function compare_models_from_thresholds(modelstocompare, modellabels, aimedmeasurementnumbers, numimages::Integer; typeofdata=:test, savefile="../experiment_data/ansdata.BSON", kwargs...)
    # Still need to debug this

    dataset = MNIST(Float32, typeofdata).features

    numexperiments = numimages * length(modelstocompare)

    results = DataFrame(threshold=Vector{Float32}(undef, numexperiments),
        fitplot=Vector{Plots.Plot}(undef, numexperiments),
        fitdata=Vector{Matrix{Float32}}(undef, numexperiments),
        fitobject=Vector{LsqFit.LsqFitResult}(undef, numexperiments),
        image=Vector{Matrix{Float32}}(undef, numexperiments),
        modelname=Vector{String}(undef, numexperiments))

    images = [dataset[:, :, rand(1:size(dataset)[3])] for i in 1:numimages]
    @threads for (i, img) in collect(enumerate(images))
        @threads for (j, model) in collect(enumerate(modelstocompare))
            returnobj = recoverythreshold_fromrandomimage(model, aimedmeasurementnumbers, img=img, savefile=nothing; kwargs...)
            results[i+(j-1)*numimages, collect(keys(returnobj))] = returnobj
            results[i+(j-1)*numimages, :modelname] = modellabels[j]
            results[i+(j-1)*numimages, :image] = img
        end
        @info i #give some idea of progress
    end

    if !isnothing(savefile)
        @save savefile aimedmeasurementnumbers results
    end

    return results
end