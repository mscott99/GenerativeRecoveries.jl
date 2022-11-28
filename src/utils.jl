import Base: *

struct fatFFTPlan
    p::AbstractFFTs.Plan
    freqs::AbstractArray{Bool}
end

function *(A::fatFFTPlan, x)
    (A.p*x)[A.freqs]
end

function inversesigmoid(y; clampmargin=1.0f-3)
    y = clamp(y, 0.0f0 + clampmargin, 1.0f0 - clampmargin)
    log(y / (1.0f0 - y))
end
"""
Run a tensor of experiments from pre-computed states.
Integer indices index both the results and the state used in the experiment
"""
function runexperimenttensor(experimentfn, experimentsetup::Vector{<:Vector}, args...; kwargs...)
    #results = Array{typeof()}(undef, map(length, experimentsetup)...) # find a way to add Parametric type
    #for index in CartesianIndices(results)
    #    results[index] = experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., args...; kwargs...)
    #end
    [experimentfn((experimentsetup[i][index[i]] for i in 1:length(experimentsetup))..., args...; kwargs...)
     for index in CartesianIndices(Tuple(map(length, experimentsetup)))]

end

