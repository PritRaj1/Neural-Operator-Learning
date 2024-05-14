module UTILS

export LpLoss, UnitGaussianNormaliser, encode, decode, log_loss

using Statistics

### Lp Norm ###
struct LpLoss
    p::Float64
end

# Constructor
function LpLoss(p::Float64)
    return LpLoss(p)
end

# Compute the loss
function (loss::LpLoss)(ŷ, y)
    return sum(abs.(ŷ .- y).^loss.p)
end


### Normaliser for zero mean and unit variance ###
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

# Log the loss to CSV
function log_loss(epoch, train_loss, test_loss)
    open("logs/loss.csv", "a") do file
        write(file, "$epoch,$train_loss,$test_loss\n")
    end
end

end
