
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