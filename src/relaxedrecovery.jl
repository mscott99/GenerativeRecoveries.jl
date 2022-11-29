#using Flux: Chain

#include("generativecompressedsensing.jl") #we need the optimise function


function _relaxedloss(measurements, A, linkstrength::AbstractFloat, networkparts::AbstractArray, fullcode::Tuple{Vararg{<:AbstractArray}})
    @assert length(fullcode) == length(networkparts) "the lengths of  fullcode  and  networkparts  must match, they are $(length(fullcode)) and $(length(networkparts))"
    linkloss = 0
    for (i, networkpart) in enumerate(networkparts[1:end-1])
        linkloss += sum(abs2.(networkpart(fullcode[i]) .- fullcode[i+1]))
    end
    mismatchloss = sum(abs2.(A * networkparts[end](fullcode[end]) .- measurements))

    mismatchloss + linkstrength * linkloss
end


function relaxed_recover(measurements, A, generativenet::Flux.Chain; optimlayers=collect(indexof(generativenet.layers)), linkstrength=1.0f0, kwargs...)

    optimloss(x, p::Tuple) = _relaxedloss(p..., x)

    netparts = AbstractArray{Chain}([])
    push!(netparts, generativenet.layers[1:optimlayers[1]] |> Chain)
    for i in 2:length(optimlayers)
        push!(netparts, generativenet.layers[optimlayers[i-1]+1:optimlayers[i]] |> Flux.Chain)
    end
    push!(netparts, generativenet.layers[optimlayers[end]+1:end] |> Chain)

    p = (measurements, A, linkstrength, netparts)

    #Get all codes between relevent layers
    codes = [randn(Float32, size(generativenet.layers[1].weight)[2])]
    for index in optimlayers
        if (generativenet.layers[index] isa Dense)
            push!(codes, randn(Float32, size(generativenet.layers[index].weight)[1]))
        elseif (generativenet.layers[index-1] isa Dense)
            push!(codes, randn(Float32, size(generativenet.layers[index].weight)[2]))
        else
            throw("Cannot find intermediate code size")
        end
    end
    codes = Tuple(codes)
    # The problem is that 

    recoveredencodings = optimise!(optimloss, p, codes; kwargs...)

    netparts[end](recoveredencodings[end])
end