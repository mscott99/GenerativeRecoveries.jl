import Base: *

struct IndexedMatrix{T,L}
    A::T
    indices::L
end

*(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

function inversesigmoid(y; clampmargin=1.0f-3)
    y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
    log(y / (1.0f0 - y))
end

function wrap_model_withreshape(model::FullVae)
    new_decoder = Chain(model.decoder..., x -> reshape(x, 28, 28))
    return FullVae(model.encoder, new_decoder)
end

function wrap_model_withsigmoid(model::FullVae)
    if model.decoder isa Chain
        lastlayer = model.decoder.layers[end]
        if lastlayer isa Dense
            new_lastlayer = Dense(lastlayer.weight, lastlayer.bias, sigmoid)
            model.decoder.layers[end] = new_lastlayer
        else
            throw("Unimplemented")
        end
    else
        throw("Unimplemented")
    end
end
"""
Run a tensor of experiments from pre-computed states.
Integer indices index both the results and the state used in the experiment
"""
function runexperimenttensor(experimentfn::Function, experimentsetup::Tuple, args...; multithread=false, kwargs...)

    #result_type = typeof(experimentfn((experimentsetup[i][1] for i in 1:length(experimentsetup))..., args...; kwargs...))
    #results = Array{result_type}(undef, length.(experimentsetup))
    if !multithread
        results = [experimentfn(element..., args...; kwargs...) for element in Iterators.product(experimentsetup...)]
    else
        results = Array{Any}(undef, length.(experimentsetup))
        indices = CartesianIndices(length.(experimentsetup))
        @threads for index in indices
            results[index] = experimentfn((experimentsetup[i][index[i]] for i in eachindex(experimentsetup))..., args...; kwargs...)
        end
    end
    # can also check if splitting up the result array with push! is sufficient


    #for (i, elt) in enumerate(Base.Iterators.product(experimentsetup...))
    #    results[i] = experimentfn(elt..., args...; kwargs...)
    #end

    results
end

