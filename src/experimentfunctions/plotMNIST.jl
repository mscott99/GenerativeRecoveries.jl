
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

    model = _setupmodel(model; presigmoid)
    decoder = model.decoder
    truesignals = _setupMNISTimagesignals(images, model; datasplit, presigmoid, inrange, kwargs...)
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    freqs = _setupfrequencies(aimedmeasurementnumbers, size(truesignals[1]))
    pdct = plan_dct(truesignals[1])

    setuplabels = (:truesignal, :frequency)
    experimentsetup = (truesignals, freqs)
    fixedsetup = (decoder, pdct, recoveryfn)
    resultdataframe = runexperimenttensor(experimentfn, experimentsetup, setuplabels, fixedsetup...; kwargs...)
    _plot_tables_ofrecoveries(resultdataframe; presigmoid) # in progress
end


function plot_MNISTrecoveryerrors(model::FullVae, samplingfunctions::Vector{<:Function}, aimedmeasurementnumbers, images, recoveryfn::Function=recoversignal; presigmoid=true, inrange=true, datasplit=:test, multithread=false, kwargs...)
    function experimentfn(truesignal, frequency, model, pdct, recoveryfn; multithread=true, kwargs...)
        if multithread
            pdct = deepcopy(pdct)
        end
        A = IndexedMatrix(pdct, frequency)
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, model; kwargs...)
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        ((:relative_error => relativeerr),)
    end

    truesignals = _setupMNISTimagesignals(images, model; datasplit, presigmoid, inrange, kwargs...)
    freqs = [frequencyfunction(aimed_m) for aimed_m in aimedmeasurementnumbers for frequencyfunction in samplingfunctions]
    # make table of frequencies and specify two labels. pass this to runexperiments. 
    _setupfrequencies(aimedmeasurementnumbers, size(truesignals[1]))


    model = _setupmodel(model; presigmoid)
    decoder = model.decoder
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    pdct = plan_dct(truesignals[1])

    setuplabels = (:truesignal, :frequency)
    experimentsetup = (truesignals, freqs)
    fixedsetup = (decoder, pdct, recoveryfn)
    resultdataframe = runexperimenttensor(experimentfn, experimentsetup, setuplabels, fixedsetup...; kwargs...)
    return resultdataframe
    plot_recovery_errors(resultdataframe)
end

function experimentgetrelative_error(truesignal, frequency, model, pdct, recoveryfn; multithread=true, kwargs...)
    #if multithread
    #     pdct = deepcopy(pdct)
    # end
    pdct = deepcopy(pdct)
    A = IndexedMatrix(pdct, frequency)
    measurements = A * truesignal
    recoveryimg = recoveryfn(measurements, A, model; kwargs...)
    relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
    Dict(:relerr => relativeerr)
end


function get_recovery_errors_tocompare_frequencysamplingalgorithms(model, images, aimed_ms, sampleevenlyfn, sampleunevenlyfn, img_size; presigmoid=true, inrange=true, datasplit=:test, multithread=false, kwargs...)
    model = _setupmodel(model; presigmoid)
    samplingfnlabels = ["even", "uneven"]
    samplingfndict = Dict("even" => sampleevenlyfn, "uneven" => sampleunevenlyfn)
    decoder = model.decoder
    truesignals = _setupMNISTimagesignals(images, model; presigmoid, inrange, datasplit, kwargs...)
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    pdct = plan_dct(truesignals[1])

    frame = allcombinations(DataFrame, (:truesignal => truesignals), (:numfrequencies => aimed_ms), (:algname => samplingfnlabels))
    transform!(frame, [:numfrequencies, :algname] => ByRow((m, algname) -> samplingfndict[algname](m, img_size)) => :sampledfrequencies)
    transform!(frame, [:truesignal, :sampledfrequencies] => ByRow((truesignal, frequency) -> experimentgetrelative_error(truesignal, frequency, decoder, pdct, recoversignal; kwargs...)) => AsTable)

    frame = combine(groupby(frame, [:numfrequencies, :algname]), :relerr, :algname, :relerr => mean => :meanrelerr, :relerr => std => :std_deviation, :relerr => (x -> 10^(mean(log10.(x)) - std(log10.(x)))) => :botuncert, :relerr => (x -> 10^(mean(log10.(x)) + std(log10.(x)))) => :topuncert)
    #plot_scatter_band(frame)
end

function plot_recovery_errors_tocompare_frequencysamplingalgorithms(model, images, aimed_ms, sampleevenlyfn, sampleunevenlyfn, img_size; presigmoid=true, inrange=true, datasplit=:test, multithread=false, kwargs...)
    frame = get_recovery_errors_tocompare_frequencysamplingalgorithms(model, images, aimed_ms, sampleevenlyfn, sampleunevenlyfn, img_size; presigmoid, inrange, datasplit, multithread, kwargs...)
    transform!(frame, :numfrequencies => (x -> x ./ 784) => :normalizednumfrequencies)

    mapmargins = mapping(:normalizednumfrequencies, :botuncert, :topuncert) * visual(AlgebraOfGraphics.Band) * visual(alpha=0.2)
    mapmeans = mapping(:normalizednumfrequencies, :meanrelerr) * visual(AlgebraOfGraphics.Lines)
    mapscatter = mapping(:normalizednumfrequencies, :relerr) * visual(AlgebraOfGraphics.Scatter)
    plt = AlgebraOfGraphics.data(frame) * (mapmargins + mapmeans + mapscatter) * mapping(color=:algname => "Sampling Type")
    re = draw(plt; axis=(; xscale=log10, yscale=log10, xlabel="Rate of Frequency Sampling", ylabel="Relative Recovery Error"))
    #plot_scatter_band(frame)
end