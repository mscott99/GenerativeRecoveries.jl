# subsampled dct
import Base: *
struct IndexedMatrix{T,L}
    A::T
    indices::L
end
*(indexedMatrix::IndexedMatrix, x::AbstractArray) = (indexedMatrix.A*x)[indexedMatrix.indices]

# getting random frequencies
function getuniformlysampledfrequencieswithreplacement(aimed_m::Integer, img_size::Tuple{Vararg{Int}}; rng=TaskLocalRNG(), kwargs...)
    rand(rng, Bernoulli(aimed_m / prod(img_size)), img_size...)
end

function samplefrequenciesuniformly(num_frequencies, img_size::Tuple{Vararg{Int}}; rng=TaskLocalRNG(), kwargs...)
    base = zeros(Bool, img_size...)
    addsamplinguniformlyatrandom!(base, num_frequencies; rng=rng, kwargs...)
    return base
end

function samplefromarray(a, num_elements; rng=Random.GLOBAL_RNG)
    picked_indices = []
    while num_elements > 0
        index = rand(1:length(a))
        if index ∉ picked_indices
            push!(picked_indices, index)
            num_elements -= 1
        end
    end
    return a[picked_indices]
end

function samplefrequenciesuniformlyanddeterministically(deterministic_m::Integer, uniform_m::Integer, img_size::Tuple{Vararg{Int}}; rng=TaskLocalRNG(), kwargs...)
    dim = length(img_size)
    num_orthants = 2^dim
    volume = deterministic_m * num_orthants
    radius = volume^(1 / dim) * gamma(dim / 2 + 1)^(1 / dim) / sqrt(π)
    frequencysamplingarray = BitArray((sum(abs2, Tuple(index) .- 0.5) ≤ radius^2 for index in CartesianIndices(img_size)))
    addsamplinguniformlyatrandom!(frequencysamplingarray, deterministic_m + uniform_m - sum(frequencysamplingarray))
    frequencysamplingarray
end

function samplefrequenciesuniformlyanddeterministically(aimed_m::Integer, img_size::Tuple{Vararg{<:Integer}}; rng=TaskLocalRNG())
    deterministic_m = round(Int, aimed_m / 2)
    uniform_m = aimed_m - deterministic_m
    samplefrequenciesuniformlyanddeterministically(deterministic_m, uniform_m, img_size; rng=rng)
end

function addsamplinguniformlyatrandom!(frequencysamplingarray::AbstractArray{<:Bool}, numtoadd::Integer; rng=TaskLocalRNG())
    indexset = CartesianIndices(size(frequencysamplingarray))
    while numtoadd != 0
        tryindex = rand(indexset)
        if frequencysamplingarray[tryindex] == 0
            frequencysamplingarray[tryindex] = 1
            numtoadd -= 1
        end
    end
end


function sampledeterministicallyfirstfrequencies(deterministic_m::Integer, img_size::Tuple{Vararg{Int}}; kwargs...)
    img_size = (5, 5, 3)
    dim = length(img_size)
    num_orthants = 2^dim
    volume = deterministic_m * num_orthants
    radius = volume^(1 / dim) * gamma(dim / 2 + 1)^(1 / dim) / sqrt(π)
    BitArray((sum(abs2, Tuple(index) .- 0.5) ≤ radius^2 for index in CartesianIndices(img_size)))
end




function sampleFourierwithoutreplacement(aimed_m, n; rng=TaskLocalRNG())
    F = Float32.(dct(diagm(ones(n)), 2))
    sampling = rand(rng, Bernoulli(aimed_m / n), n)
    true_m = sum(sampling)
    F[sampling, :] * sqrt(n / true_m)
    #return get_true_m ? (true_m, normalized_F) : normalized_F # normalize it
end

function sampleFourierwithoutreplacement(aimed_m, n, returntruem)
    F = Float32.(dct(diagm(ones(n)), 2))
    sampling = rand(Bernoulli(aimed_m / n), n)
    true_m = sum(sampling)
    normalized_F = F[sampling, :] * sqrt(n / true_m)
    (true_m, normalized_F)
end

function fullfourier(dim)
    dct(diagm(ones(dim)), 2)
end