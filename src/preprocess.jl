"""Pre-process models for recovery experiments"""
function setupmodelforrecoveryexperiment(model::FullVae, presigmoid=true; kwargs...)
    return setupmodelsforrecoveryexperiment([model], presigmoid; kwargs...)[1]
end


"""Pre-process models for recovery experiments"""
function setupmodelsforrecoveryexperiment(models::AbstractArray{<:FullVae}, presigmoid=true; kwargs...)
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

function setupmodelsforrecoveryexperiment(model::FullVae; kwargs...)
    setupmodelsforrecoveryexperiment([model]; kwargs...)
end

getcelebafilename(i::Integer) = lpad(i, 6, "0") * ".jpg"

function getCELEBAdataset(datasplit=:all)
    train_test_split = 0.7
    dataset = FunctionalFileDataset("/Users/matthewscott/.julia/datadeps/CELEBA/img_align_celeba/", getcelebafilename)
    if datasplit in (:train, :test)
        return FunctionalSubDataset(dataset, datasplit, train_test_split)
    else
        return dataset
    end
end

function setupCELEBAimagesignals(numImages::Int; datasplit=:test, rng=TaskLocalRNG(), kwargs...)
    dataset = getCELEBAdataset(:test)
    images = samplefromarray(dataset, numImages, rng=rng)
    (x -> float32.(x)).(permutedims.(channelview.(images), [(2, 3, 1)]))
end

"Plot CelebA image"
function showCELEBAimage(img)
    colorview(RGB, permutedims(img, (3, 1, 2)))
end

"""
Standardise the image inputs
"""
function setupMNISTimagesignals(numbers::Vector{<:Integer}, fullmodel::FullVae; datasplit=:test, presigmoid=true, inrange=true, rng=TaskLocalRNG(), kwargs...)
    images = getMNISTbynumbers(numbers; datasplit)
    if inrange
        return fullmodel(images, 10, rng=rng)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        #return (in -> inversesigmoid.(in)).(images) # double broadcast
        return inversesigmoid.(images) # double broadcast
    else
        return images
    end
end

function getMNISTbynumbers(numbers::Vector{<:Integer}; datasplit=:test, rng=TaskLocalRNG(), kwargs...)
    data = MNIST(Float32, datasplit)
    cat([reshape(data.features[:, :, data.targets.==number][:, :, rand(rng, 1:size(data.features[:, :, data.targets.==number])[end])], 28, 28, 1) for number in numbers]..., dims=4)
end

function setupMNISTimagesignals(number::Integer, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    setupMNISTimagesignals([number], model; datasplit, presigmoid, inrange, kwargs...)
end

function setupMNISTimagesignals(images::AbstractArray{<:Matrix}, model::FullVae; datasplit=:test, presigmoid=true, inrange=true, kwargs...)
    if inrange
        return model(images, 10)
    elseif presigmoid # not in range and presigmoid; only then do we need to invert the signals. Otherwise the modified decoder takes care of it.
        return inversesigmoid.(images) # double broadcast
    else
        return images
    end
end

function setupfrequencies(pre_sampled_frequencies::Vector{<:AbstractArray{<:Bool}}, img_size, batch_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    pre_sampled_frequencies
end

function setupfrequencies(pre_sampled_frequencies::AbstractArray{<:Bool}, img_size, batch_size; rng=TaskLocalRNG(), kwargs...)
    # identity
    [pre_sampled_frequencies]
end

function setupfrequencies(aimed_ms::Vector{<:Integer}, input_size::Tuple; rng=TaskLocalRNG(), samplingfn=getuniformlysampledfrequencieswithreplacement, kwargs...)
    if length(input_size) == 3 # make one filter per aimed m
        [samplingfn(aimed_m, input_size; rng, kwargs...) for aimed_m in aimed_ms]
    elseif lengt(input_size) == 4 # make one filter per image in the batch per aimed m
        [cat((samplingfn(aimed_m, input_size[1:3]; rng, kwargs...) for _ in 1:input_size[4])..., dims=4) for aimed_m in aimed_ms]
    end
end

function setupfrequencies(aimed_m::Integer, img_size::Tuple{<:Integer,<:Integer}, batch_size; rng=TaskLocalRNG(), kwargs...)
    setupfrequencies([aimed_m], img_size, batch_size; rng=rng)
end

function setupfrequencies(p::AbstractArray, ; kwargs...)
    [rand.(Bernoulli.(p))]
end
