import Base: *

struct IndexedMatrix{T,L}
    A::T
    indices::L
end

import Base: *

*(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

struct ParallelMatrix{T}
    A::AbstractArray{T}
end
ParallelMatrix(A) = ParallelMatrix([deepcopy(A) for i in 1:nthreads()])

*(parallelMatrix::ParallelMatrix, x::AbstractArray) = parallelMatrix.A[threadid()] * x


# struct FatFFTPlan{T<:AbstractFFTs.Plan,F<:AbstractArray{Bool}} <: AbstractFFTs.Plan{T}
#     p::T
#     freqs::F
# end

# *(A::FatFFTPlan, x::AbstractArray) = (A.p*x)[A.freqs]





# struct ParallelFFTPlan{T<:AbstractFFTs.Plan} <: AbstractFFTs.Plan{T}
#     pdct::AbstractArray{T}
# end

# ParallelFFTPlan(pdct::AbstractFFTs.Plan) = ParallelFFTPlan([deepcopy(pdct) for i in 1:nthreads()])

# *(a::ParallelFFTPlan, x::AbstractArray) = a.pdct[threadid()] * x


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
    # currently computes the first one twice, alternatively could do array comprehension but then without multithreading)
    #result_type = typeof(experimentfn((experimentsetup[i][1] for i in 1:length(experimentsetup))..., args...; kwargs...))
    #results = Array{result_type}(undef, length.(experimentsetup))

    #indices = collect(CartesianIndices(length.(experimentsetup)))
    # reshape seems to be necessary for multithreading to work

    #for index in indices
    #    results[index] = experimentfn(local_setup..., localargs...; kwargs...)
    #end

    # working but slow
    # @threads for index in indices
    #     local_setup = deepcopy([experimentsetup[i][index[i]] for i in 1:length(experimentsetup)])
    #     localargs = deepcopy(args)
    #     results[index] = experimentfn(local_setup..., localargs...; kwargs...)
    # end
    if !multithread

        #this formulation does not work for multi-threaded because of the need to iterate over indices
        results = [experimentfn(element..., args...; kwargs...) for element in Iterators.product(experimentsetup...)]

        # marginally faster (1/50)
        #results = []
        #irst = true
        # for (index, element) in enumerate(Iterators.product(experimentsetup...))
        #     if first
        #         firstelt = experimentfn(element..., args...; kwargs...)
        #         results = Array{typeof(firstelt)}(undef, length.(experimentsetup))
        #         results[index] = firstelt
        #         first = false
        #     else
        #         results[index] = experimentfn(element..., args...; kwargs...)
        #     end
        # end
    else
        # currently slower for some reason

        result_type = typeof(experimentfn((experimentsetup[i][1] for i in 1:length(experimentsetup))..., args...; kwargs...))
        results = Array{result_type}(undef, length.(experimentsetup))
        # results = Array{Any}(undef, length.(experimentsetup))
        # Array{Any}(undef, length.(experimentsetup))

        #It looks like we need a distinct planned dct for each thread
        # copied_setup = [deepcopy(experimentsetup) for i in 1:nthreads()]

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

