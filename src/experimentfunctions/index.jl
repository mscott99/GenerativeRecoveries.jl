
#include("VaeModels.jl")
#using .VaeModels

#include("generativecompressedsensing.jl")
#include("plottingfunctions.jl")

"Makes the true signal of the correct type and shape"
function _preprocess_MNIST_truesignal(img, VAE::Union{FullVae,ComposedFunction}, presigmoid, inrange; rng=TaskLocalRNG())

    function inversesigmoid(y; clampmargin=1.0f-3)
        y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
        log(y / (1.0f0 - y))
    end

    truesignal = plottedtruesignal = img

    if inrange
        truesignal = VAE(img, 100, rng=rng)
        plottedtruesignal = presigmoid ? sigmoid(truesignal) : truesignal
    elseif presigmoid
        truesignal = inversesigmoid.(img)
    end

    return truesignal, plottedtruesignal
end


function imagesfromnumbers(numbers::AbstractArray{<:Integer}, typeofdata; rng=TaskLocalRNG())
    data = MNIST(Float32, typeofdata)
    images = []
    for number in numbers
        numberset = data.features[:, :, data.targets.==number]
        push!(images, numberset[:, :, rand(rng, 1:size(numberset)[end])])
    end
    convert(AbstractArray{typeof(images[1])}, images)
end


function imagesfromnumbers(numbers::Integer, typeofdata; rng=TaskLocalRNG())
    data = MNIST(Float32, typeofdata)
    numberset = data.features[:, :, data.targets.==numbers]
    [numberset[:, :, rand(rng, 1:size(numberset)[end])]]
end

function setupExperimentDecoderandImages(VAE::FullVae, images::Vector{<:AbstractArray}, presigmoid, inrange)
    if !presigmoid #preprocess the models
        lastlayer = VAE.decoder.layers[end]
        newlastlayer = Dense(lastlayer.weight, false, sigmoid)
        decoder = Chain(VAE.decoder.layers[1:end-1]..., newlastlayer)
        #VAE = FullVae(VAE.encoder, Chain(VAE.decoder.layers[1:end-1]..., newlastlayer))
    end
    true_signals = similar(images)
    truesignals_toplot = similar(images)
    for (i, image) in enumerate(images)
        true_signals[i], truesignals_toplot[i] = _preprocess_MNIST_truesignal(image, VAE, presigmoid, inrange)
    end

    return decoder, true_signals, truesignals_toplot
end

function setupExperimentDecoderandImages(VAE::FullVae, image::AbstractArray, presigmoid, inrange)
    if !presigmoid #preprocess the models
        lastlayer = VAE.decoder.layers[end]
        newlastlayer = Dense(lastlayer.weight, false, sigmoid)
        decoder = Chain(VAE.decoder.layers[1:end-1]..., newlastlayer)
        #VAE = FullVae(VAE.encoder, Chain(VAE.decoder.layers[1:end-1]..., newlastlayer))
    end
    true_signal, truesignal_toplot = _preprocess_MNIST_truesignal(image, VAE, presigmoid, inrange)

    return decoder, true_signal, truesignal_toplot
end



getlayerdims(ChainDecoder::Flux.Chain{<:Tuple{Vararg{Dense}}}) =
    vcat([size(layer.weight)[2] for layer in ChainDecoder.layers], [size(ChainDecoder.layers[end].weight)[1]])


#used to choose measurement number in a smart way

logrange(low_meas, high_meas, num_meas) = convert.(Int, floor.(exp.(LinRange(log(low_meas), log(high_meas), num_meas))))
#collect(0:10:220)
#@time recoverythreshold_fromrandomimage(model, model.decoder, collect(0:10:40), 16, 28^2)

include("./plotMNIST.jl")
include("./plotrecoveryerrors.jl")
include("./thresholdRecoveryMeasurements.jl")



