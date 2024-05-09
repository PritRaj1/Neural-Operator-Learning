module Normalisers

export UnitGaussianNormaliser, encode, decode

using Statistics

struct UnitGaussianNormaliser{T<:AbstractFloat}
    mean::T
    std::T
    eps::T
end

# Constructor, characterises the distribution of the data
function UnitGaussianNormaliser(x::AbstractArray{T}, eps::T=1e-5) where {T<:AbstractFloat}
    mean = Statistics.mean(x)
    std = Statistics.std(x)
    return UnitGaussianNormaliser(mean, std, eps)
end

# Normalise to zero mean and unit variance
function encode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return (x .- normaliser.mean) ./ (normaliser.std .+ normaliser.eps)
end

# Denormalise
function decode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return x .* (normaliser.std .+ normaliser.eps) .+ normaliser.mean
end

end