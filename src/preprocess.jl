"""Pre-process models for recovery experiments"""
function _setupmodel(model::FullVae, presigmoid=true; kwargs...)
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
function _setupmodels(models::AbstractArray{<:FullVae}, presigmoid=true; kwargs...)
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

function _setupmodels(model::FullVae; kwargs...)
    _setupmodels([model]; kwargs...)
end

getcelebafilename(i::Integer) = lpad(i, 6, "0") * ".jpg"

function _getCELEBAdataset(datasplit=:all)
    train_test_split = 0.7
    dataset = FunctionalFileDataset("/Users/matthewscott/.julia/datadeps/CELEBA/img_align_celeba/", getcelebafilename)
    if datasplit in (:train, :test)
        return FunctionalSubDataset(dataset, datasplit, train_test_split)
    else
        return dataset
    end
end

function setupCELEBAimagesignals(numImages::Int; datasplit=:test, rng=TaskLocalRNG(), kwargs...)
    dataset = _getCELEBAdataset(:test)
    images = samplefromarray(dataset, numImages, rng=rng)
    (x -> float32.(x)).(permutedims.(channelview.(images), [(2, 3, 1)]))
end


"""
Standardise the image inputs
"""
function _setupMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::FullVae; datasplit=:test, presigmoid=true, inrange=true, rng=TaskLocalRNG(), kwargs...)
    images = _setupMNISTimagesignals(numbers; datasplit)
    if inrange
        return fullmodel.(images, 10, rng=rng)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        return (in -> inversesigmoid.(in)).(images) # double broadcast
    else
        return images
    end
end

function _setupMNISTimagesignals(numbers::Vector{<:Integer}; datasplit=:test, rng=TaskLocalRNG(), kwargs...)
    data = MNIST(Float32, datasplit)
    [data.features[:, :, data.targets.==number][:, :, rand(rng, 1:size(data.features[:, :, data.targets.==number])[end])] for number in numbers]
end

function _setupMNISTimagesignals(number::Integer, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    _setupMNISTimagesignals([number], model; datasplit, presigmoid, inrange, kwargs...)
end

function _setupMNISTimagesignals(images::AbstractArray{<:Matrix}, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    if inrange
        return model.(images, 10)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        return (in -> inversesigmoid.(in)).(images) # double broadcast
    else
        return images
    end
end

function _setupfrequencies(pre_sampled_frequencies::Vector{<:AbstractArray{<:Bool}}, img_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    pre_sampled_frequencies
end

function _setupfrequencies(pre_sampled_frequencies::AbstractArray{<:Bool}, img_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    [pre_sampled_frequencies]
end

function _setupfrequencies(aimed_ms::Vector{<:Integer}, img_size::Tuple; rng=TaskLocalRNG(), samplingfn=getuniformlysampledfrequencieswithreplacement, kwargs...)
    [samplingfn(aimed_m, img_size; rng, kwargs...) for aimed_m in aimed_ms]
end

function _setupfrequencies(aimed_m::Integer, img_size::Tuple{<:Integer,<:Integer}; rng=TaskLocalRNG(), kwargs...)
    _setupfrequencies([aimed_m], img_size; rng=rng)
end

function _setupfrequencies(p::AbstractArray, ; kwargs...)
    [rand.(Bernoulli.(p))]
end
