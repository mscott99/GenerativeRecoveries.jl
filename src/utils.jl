import Base: *

struct fatFFTPlan
    p::AbstractFFTs.Plan
    freqs::AbstractArray{Bool}
end

function *(A::fatFFTPlan, x)
    (A.p*x)[A.freqs]
end