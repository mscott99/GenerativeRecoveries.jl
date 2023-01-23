"""
    optimise(init_z, loss, opt, tolerance, [out_toggle = 0,][max_iter = 1_000_000])

    Optimization that stops when the gradient is small enough

    loss takes z and p as arguments.
"""
function optimise!(loss, p, z; opt=Adam(5.0f-2), tolerance=1.0f-2, out_toggle=1e2, max_iter=10_000, tblogdir=nothing, kwargs...)
    tol2 = tolerance^2
    usingtb = !isnothing(tblogdir)
    logger = usingtb ? TBLogger(tblogdir) : current_logger()

    #ps = typeof(z) <: AbstractArray{<:AbstractArray} ? params(z...) : params(z)
    opt_state = Flux.setup(opt, z)
    #ps = params(z)
    iter = 1
    succerror = 0.0f0
    with_logger(logger) do
        while true
            if iter > max_iter
                #@warn "Max num. iterations reached"
                return nothing
            end
            grads = gradient(loss, z, p)[1]
            #grads = gradient(() -> loss(z, p), ps) #loss cannot have any arguments

            opt_state, z = update!(opt_state, z, grads)

            #if typeof(z) <: Tuple
            #    succerror = sum((sum(abs2, grads[elt]) for elt in z))
            #else
            succerror = sum(abs2, grads) / size(grads, 4)
            #end

            if usingtb && out_toggle != 0 && iter % out_toggle == 0
                @info "recovery optimization step" iter grad_size = sqrt(succerror) lossval = sqrt(loss(z))
            end
            if succerror < tol2

                break
            end
            iter += 1
        end
    end
    @debug "final stats" final_gradient_size = sqrt(succerror) iter thread = threadid()
    return z
end

function optimise_new!(getgradients, p, z; opt=Adam(5.0f-2), tolerance=1.0f-2, out_toggle=1e2, max_iter=10_000, tblogdir=nothing, kwargs...)
    tol2 = tolerance^2
    usingtb = !isnothing(tblogdir)
    logger = usingtb ? TBLogger(tblogdir) : current_logger()

    #ps = typeof(z) <: AbstractArray{<:AbstractArray} ? params(z...) : params(z)
    opt_state = Flux.setup(opt, z)
    #ps = params(z)
    iter = 1
    succerror = 0.0f0
    with_logger(logger) do
        while true
            if iter > max_iter
                @warn "Max num. iterations reached"
                return missing
            end

            grads = getgradients(z, p)
            #grads = gradient(() -> loss(z, p), ps) #loss cannot have any arguments

            opt_state, z = update!(opt_state, ps, grads)

            if typeof(z) <: Tuple
                succerror = sum((sum(abs2, grads[elt]) for elt in z))
            else
                succerror = sum(abs2, grads[z])
            end

            if usingtb && out_toggle != 0 && iter % out_toggle == 0
                @info "recovery optimization step" iter grad_size = sqrt(succerror) lossval = sqrt(loss(z))
            end
            if succerror < tol2

                break
            end
            iter += 1
        end
    end
    @debug "final stats" final_gradient_size = sqrt(succerror) iter thread = threadid()
    return z
end

#function recoversignal(measurement, A, decoder; init_code=randn(Float32, size(decoder.layers[1].weight)[2]) ./ size(decoder.layers[1].weight)[2], kwargs...)
#recover single signal
#    @debug "Starting Image Recovery"
#    loss(x, p::Tuple) = sum(abs2, A * p[1](x) - p[2])
#    p = (decoder, measurement)
#    decoder(optimise!(loss, p, init_code; kwargs...))
#end

function recoversignal(measurements, A, decoder, codesize; init_code=randn(Float32, codesize) ./ prod(codesize[1:3]), kwargs...)
    #recover a batch of signals
    #current approach: split the batch and recover each one.
    @debug "Starting Image Recovery"
    #split batch into smaller ones
    #results = eltype()
    #for i in size(measurements, 2)
    #    results[:,:,:,i] = recoversignal(measurements[:,i], A, decoder, codesize; )
    #end

    #arrayed_data = [x[:, :, :, i:i] for i in size(x)[4]]
    #arrayed_measurements = [measurements[:,i] for i in size(measurements)[end]]

    "We need this because Zygote does not handle iterating over slices well, so we use broadcast"
    #function getgrads(x, p::Tuple)
    #    arrayed_data = [x[:, :, :, i:i] for i in size(x)[4]]
    #    arrayed_measurements = [measurements[:,i] for i in size(measurements)[end]]
    #    grads = gradient(arrayed_data -> sum(sum.([abs2], [A] .* p[1].(arrayed_data) .- ), arrayed_data)
    #    cat(grads, length(size(x)))
    #endx
    function loss(x, p)
        sum(sum.([abs2], [A] .* Flux.unstack(p[1](x), dims=4) .- Flux.unstack(p[2], dims=2)))
    end
    #loss(x, p::Tuple) = sum(abs2, measurementfn(p[1](x)) - p[2])
    p = (decoder, measurements)
    decoder(optimise!(loss, p, init_code; tolerance=1.0f2, max_iter=3000, kwargs...))
end

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