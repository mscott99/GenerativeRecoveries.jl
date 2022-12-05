
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