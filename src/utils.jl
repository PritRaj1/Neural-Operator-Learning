module UTILS

export LpLoss, loss_fcn, UnitGaussianNormaliser, encode, decode, log_loss, get_grid

using Statistics
using CUDA, KernelAbstractions, Tullio
using Flux

p = parse(Float32, get(ENV, "p", "2.0"))
nx, ny = 32, 32

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

# Creates grids for spectral convolutions (x, y, 1, batch_size) -> (3, x, y, batch_size)
X = Float32.(range(0,1,nx))
Y = Float32.(range(0,1,ny))
X = reshape(X, 1, nx, 1, 1)
Y = reshape(Y, 1, 1, ny, 1)

function get_grid(x)
    batch_size = size(x, 4)
    gridx = repeat(X, 1, 1, ny, batch_size)
    gridy = repeat(Y, 1, nx, 1, batch_size)
    grid = cat(gridx, gridy, dims=1) |> gpu
    x_reshaped = @tullio y[c, w, h, b] := x[w, h, c, b]
    return vcat(x_reshaped, grid)
    
end
end
