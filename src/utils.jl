"""
Run a tensor of experiments from pre-computed states.
Integer indices index both the results and the state used in the experiment
"""
function runexperimenttensor(experimentfn::Function, experimentsetup::Tuple, setupsymbols::Tuple, args...; include_values=true, kwargs...)::DataFrame
    multithread = false # multithreading not functional right now; first have to check type safety
    #result_type = typeof(experimentfn((experimentsetup[i][1] for i in 1:length(experimentsetup))..., args...; kwargs...))
    #results = Array{result_type}(undef, length.(experimentsetup))
    if !multithread   # still needs testing
        first = true
        df = nothing # initialize symbol for scope purposes
        setupsymbolsindex = (Symbol(String(key) * "_index") for key in setupsymbols)
        iterators = (enumerate(setuparray) for setuparray in experimentsetup)
        for specificsetup in Iterators.product(iterators...) # iterator over dict
            #specific setup is a tuple of dict pairs
            elementsetup = (it[2] for it in specificsetup)
            elementsetupindices = (it[1] for it in specificsetup)
            experimentresults = experimentfn(elementsetup..., args...; kwargs...) # should be a tuple of pairs

            namedsetupindices = ((name => setupargument) for (name, setupargument) in zip(setupsymbolsindex, elementsetupindices))
            if include_values
                namedsetup = ((name => setupargument) for (name, setupargument) in zip(setupsymbols, elementsetup))
                df_entry = Dict((namedsetup..., namedsetupindices..., experimentresults...))
            else
                df_entry = Dict((namedsetupindices..., experimentresults...))
            end
            if first
                init_types = Dict((name => typeof(entry)[]) for (name, entry) in pairs(df_entry))
                df = DataFrame(pairs(init_types)...)
                push!(df, df_entry)
                first = false
            else
                push!(df, df_entry)
            end
        end
        return df::DataFrame
    else
        throw("Not Implemented") # must first implement dataframe pre-allocation 
        results = Array{Any}(undef, length.(experimentsetup))
        indices = CartesianIndices(length.(experimentsetup))
        @threads for index in indices
            results[index] = experimentfn((experimentsetup[i][index[i]] for i in eachindex(experimentsetup))..., args...; multithread=multithread, kwargs...) # experimentfn should return a dictionary
        end
    end

    # initialize the dataframe with the correct types
    # this code is not accessible
    #alltypes = Dict()
    #for (label, setuparray) in zip(setuplabels, experimentsetup)
    #alltypes[label] = typeof(setuparray[1])[]
    #alltypes[label*"_index"] = Int[]
    #end

    #resultinitdict = Dict((k, typeof(v)[]) for (k, v) in results[1])
    #merge!(alltypes, resultinitdict)

    #frame = DataFrame(alltypes)

    ## fill the dataframe
    #for index in CartesianIndices(results)
    #entries = Dict()
    #for i in eachindex(experimentsetup)
    #entries[setuplabels[i]] = experimentsetup[i][index[i]]
    #entries[setuplabels[i]*"_index"] = index[i]
    #end
    #merge!(entries, results[index])
    #push!(frame, entries)
    #end

    #frame
end




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
