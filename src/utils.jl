module UTILS

export LpLoss, loss_fcn, UnitGaussianNormaliser, encode, decode, log_loss, get_grid, complexGeLU

using Statistics, SpecialFunctions

p = parse(Float32, get(ENV, "p", "2.0"))

function loss_fcn(m, x, y)
    return sum(abs.(m(x) .- y).^p)
end

eps = Float32(1e-5)

### Normaliser for zero mean and unit variance ###
struct UnitGaussianNormaliser{T<:AbstractFloat}
    μ::T
    σ::T
    ε::T
end

# Constructor, characterises the distribution of the data, takes 3D array
function createNormaliser(x::AbstractArray)
    data_mean = Statistics.mean(x)
    data_std = Statistics.std(x)
    return UnitGaussianNormaliser(data_mean, data_std, eps)
end

# Normalise to zero mean and unit variance
function encode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return (x .- normaliser.μ) ./ (normaliser.σ .+ normaliser.ε)
end

# Denormalise
function decode(normaliser::UnitGaussianNormaliser, x::AbstractArray)
    return x .* (normaliser.σ .+ normaliser.ε) .+ normaliser.μ
end

# Log the loss to CSV
function log_loss(epoch, train_loss, test_loss, model_name)
    open("logs/$model_name.csv", "a") do file
        write(file, "$epoch,$train_loss,$test_loss\n")
    end
end

# Creates channels for spectral convolutions (x, y, 1, batch_size) -> (3, x, y, batch_size)
function get_grid(x)
    nx, ny, _, batch_size = size(x)
    X = [Float32.(x) for x in range(0, stop=1, length=nx)]
    Y = [Float32.(y) for y in range(0, stop=1, length=ny)]

    gridx = repeat(reshape(X, nx, 1, 1, 1), 1, ny, 1, batch_size)
    gridy = repeat(reshape(Y, 1, ny, 1, 1), nx, 1, 1, batch_size)

    grid = cat(x, gridx, gridy, dims=3)

    return permutedims(grid, [3, 1, 2, 4])
end
end


