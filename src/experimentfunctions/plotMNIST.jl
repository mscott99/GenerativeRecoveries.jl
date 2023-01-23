"""
Plot a matrix of recovery images by number for different measurement numbers
The VAE and VAE decoder should never have a final activation
VAE can be given as nothing if "inrange=false" is given.
"""
function plot_MNISTrecoveries(model::FullVae, aimedmeasurementnumbers, images, recoveryfn::Function=recoversignal; presigmoid=true, inrange=true, datasplit=:test, multithread=false, dct=false, kwargs...)

    function experimentfn(truesignal, frequency, model, pdct, recoveryfn; multithread=true, dct=false, kwargs...)
        if multithread
            pdct = deepcopy(pdct)
        end
        A = dct ? IndexedMatrix(pdct, frequency) : ComplexIndexedMatrix(pdct, frequency)
        measurements = A * truesignal
        recoveryimg = recoveryfn(measurements, A, model; kwargs...)
        relativeerr = norm(recoveryimg .- truesignal) / norm(truesignal)
        Dict(:recoveredsignal => recoveryimg, :relativeerror => relativeerr, :numfrequencies => sum(frequency))
    end
    model = setupmodelforrecoveryexperiment(model; presigmoid)
    decoder = model.decoder
    truesignals = setupMNISTimagesignals(images, model; datasplit, presigmoid, inrange, kwargs...)
    sampleimg = truesignals[:, :, :, 1]
    plansampleimg = sampleimg
    #plansampleimg = length(size(sampleimg)) == 4 ? sampleimg[:, :, :, 1] : sampleimg
    freqs = setupfrequencies(aimedmeasurementnumbers, size(sampleimg); dct=dct, kwargs...)
    pdct = dct ? plan_dct(plansampleimg) : plan_fft(plansampleimg)
    measurementmatrices = dct ? [IndexedMatrix(pdct, freq) for freq in freqs] : [ComplexIndexedMatrix(pdct, freq) for freq in freqs]
    #experimentsetup = ((:truesignal_index => eachindex(truesignals)), (:frequency_index => eachindex(freqs)))
    #experimentsetup = ((:frequency_index => eachindex(freqs)))

    #resultdataframe = allcombinations(DataFrame, experimentsetup...)
    resultdataframe = DataFrame(:frequencyindex => eachindex(freqs), :truesignals => Ref(truesignals))
    #transform!(resultdataframe, :truesignal_index => ByRow(ind -> truesignals[:,:,:,ind]) => :truesignal)
    codesize = Flux.outputsize(Flux.Chain(model.encoder.encoderbody, model.encoder.splitedlogvar), size(truesignals)) #generalize this: either encode this into the model as an attribute, or figure it out from weights
    @eachrow! resultdataframe begin
        A = measurementmatrices[:frequencyindex]
        #measurements = subsampledlinearmeasurement(:truesignal, pdct, freq; dct)
        y = Flux.stack([A] .* Flux.unstack(truesignals, dims=4), dims=2)
        @newcol :recoveredsignals::Vector{typeof(truesignals)}
        :recoveredsignals = recoveryfn(y, A, decoder, codesize; kwargs...)
        @newcol :relative_errors::Vector{Vector{Float32}}
        :relative_errors = reshape(sqrt.(sum(abs2, :recoveredsignals .- truesignals, dims=1:3) ./ sum(abs2, truesignals, dims=1:3)), :)
        @newcol :selectedfreqs::Vector{typeof(sampleimg)}
        :selectedfreqs = freqs[:frequencyindex]
        @newcol :numfrequencies::Vector{Int}
        :numfrequencies = sum(freqs[:frequencyindex]) # Int
    end
    #transform!(resultdataframe, [:truesignal, :frequency_index] => ByRow((truesignal, frequency_index) -> experimentfn(truesignal, freqs[frequency_index], decoder, pdct, recoveryfn; dct, kwargs...)) => AsTable)
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

    truesignals = setupMNISTimagesignals(images, model; datasplit, presigmoid, inrange, kwargs...)
    signalsize = size(truesignals[1])
    freqs = [frequencyfunction(aimed_m, signalsize) for aimed_m in aimedmeasurementnumbers for frequencyfunction in samplingfunctions]
    # make table of frequencies and specify two labels. pass this to runexperiments. 
    setupfrequencies(aimedmeasurementnumbers, size(truesignals[1]))


    model = setupmodelforrecoveryexperiment(model; presigmoid)
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

function _experimentgetrelative_error(truesignal, frequency, model, pdct, recoveryfn=recoversignal; multithread=true, kwargs...)
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
    model = setupmodelforrecoveryexperiment(model; presigmoid)
    samplingfnlabels = ["even", "uneven"]
    samplingfndict = Dict("even" => sampleevenlyfn, "uneven" => sampleunevenlyfn)
    decoder = model.decoder
    truesignals = setupMNISTimagesignals(images, model; presigmoid, inrange, datasplit, kwargs...)
    # not well adapted to the in range case with many models; for that we need a notion of dependence when running experiments. Probably a callback.
    pdct = plan_dct(truesignals[1])

    frame = allcombinations(DataFrame, (:truesignal => truesignals), (:numfrequencies => aimed_ms), (:algname => samplingfnlabels))
    transform!(frame, [:numfrequencies, :algname] => ByRow((m, algname) -> samplingfndict[algname](m, img_size)) => :sampledfrequencies)
    transform!(frame, [:truesignal, :sampledfrequencies] => ByRow((truesignal, frequency) -> _experimentgetrelative_error(truesignal, frequency, decoder, pdct, recoversignal; kwargs...)) => AsTable)

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






