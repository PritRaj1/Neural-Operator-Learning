module FNO_layers

export SpectralConv2d, MLP

using FFTW: fft, ifft
using Flux: Conv
using SpecialFunctions: erf
using ConfParser
using NNlib

conf = ConfParse("../../FNO_config.ini")
parse_conf!(conf)

modes1 = parse(Int, retrieve(conf, "Architecture", "modes1"))
modes2 = parse(Int, retrieve(conf, "Architecture", "modes2"))
activation = retrieve(conf, "Architecture", "activation")

# Activation mapping
act_fcn = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu
)[activation]

struct SpectralConv2d{T}
    w1::T
    w2::T
    in_channels::Int64
    out_channels::Int64
end

function SpectralConv2d(in_channels::Int, out_channels::Int)
    scale = 1 / (in_channels * out_channels)
    weights1 = scale * randn(ComplexF32, in_channels, out_channels, modes1, modes2)
    weights2 = scale * randn(ComplexF32, in_channels, out_channels,  modes1, modes2)
    return SpectralConv2d(weights1, weights2, in_channels, out_channels)
end

function compl_mul2d(input, weights)

    # (b, in, x, y), (in, out, x, y) -> (b, out, x, y) 
    output = zeros(ComplexF32, size(input, 1), size(weights, 2), size(input, 3), size(input, 4))
    for i in 1:size(input, 1)
        for j in 1:size(weights, 2)
            output[i, j, :, :] = sum(input[i, :, :, :] .* weights[:, j, :, :], dims=1)
        end
    end
    return output
end

function (m::SpectralConv2d)(x)

    # Fourier transform
    x_FT = fft(x, [1, 2])
    x_FT = permutedims(x_FT, [4, 3, 1, 2])

    # Multiply relevant Fourier modes
    out_FT = zeros(ComplexF32, size(x_FT, 1), m.out_channels, size(x_FT, 3), size(x_FT, 4))
    out_FT[:, :, 1:modes1, 1:modes2] = compl_mul2d(x_FT[:, :, 1:modes1, 1:modes2], m.w1)
    out_FT[:, :, end-modes1+1:end, 1:modes2] = compl_mul2d(x_FT[:, :, end-modes1+1:end, 1:modes2], m.w2)

    # Inverse fourier transform
    out_FT = permutedims(out_FT, [3, 4, 2, 1])
    return real(ifft(out_FT, [1, 2]))
end

struct MLP
    conv1
    conv2
end

function MLP(in_channels::Int64, out_channels::Int64, hidden_channels::Int64)
    mlp1 = Conv((1, 1), in_channels => hidden_channels)
    mlp2 = Conv((1, 1), hidden_channels => out_channels)
    return MLP(mlp1, mlp2)
end

function (m::MLP)(x)
    x = m.conv1(x)
    x = act_fcn(x)
    return m.conv2(x)
end

end