module Layers

export SpectralConv2d, MLP

using Flux
using Flux: rfft, irfft, Conv, gelu

struct SpectralConv2d
    in_channels::Int
    out_channels::Int
    modes1::Int
    modes2::Int
    scale::Float32
    weights1::Matrix{ComplexF32}
    weights2::Matrix{ComplexF32}
end

function SpectralConv2d(in_channels, out_channels, modes1, modes2)
    scale = 1 / (in_channels * out_channels)
    weights1 = scale * ComplexF32.(rand(Float32, in_channels, out_channels, modes1, modes2))
    weights2 = scale * ComplexF32.(rand(Float32, in_channels, out_channels, modes1, modes2))
    return SpectralConv2d(in_channels, out_channels, modes1, modes2, scale, weights1, weights2)
end

# Complex multiplication
function compl_mul2d(input, weights)
    return sum(input .* weights, dims=2)
end

@funcs SpectralConv2d (layer::SpectralConv2d)(x) = begin
    batchsize = size(x, 1)
    x_ft = rfft(x, 2)

    out_ft = zeros(ComplexF32, batchsize, layer.out_channels, size(x, 2), size(x, 3) ÷ 2 + 1)
    out_ft[:, :, 1:layer.modes1, 1:layer.modes2] =
        compl_mul2d(x_ft[:, :, 1:layer.modes1, 1:layer.modes2], layer.weights1)
    out_ft[:, :, end-layer.modes1+1:end, 1:layer.modes2] =
        compl_mul2d(x_ft[:, :, end-layer.modes1+1:end, 1:layer.modes2], layer.weights2)

    return irfft(out_ft, size(x, 2:3), 2)
end

struct MLP
    mlp1::Conv
    mlp2::Conv
    act
end

function MLP(in_channels, out_channels, mid_channels)
    mlp1 = Conv((1, 1), in_channels => mid_channels)
    mlp2 = Conv((1, 1), mid_channels => out_channels)
    act = gelu
    return MLP(mlp1, mlp2, act)
end

function (layer::MLP)(x)
    x = layer.mlp1(x)
    x = layer.act(x)
    x = layer.mlp2(x)
    return x
end
    
end
