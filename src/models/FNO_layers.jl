module FNO_layers

export SpectralConv2d, MLP

using Flux: Conv, Dense
using Flux
using ConfParser
using NNlib
using CUDA, CUDAKernels, KernelAbstractions
using CUDA.CUFFT: rfft, irfft
using Tullio

conf = ConfParse("../../FNO_config.ini")
parse_conf!(conf)

modes1 = parse(Int, retrieve(conf, "Architecture", "modes1"))
modes2 = parse(Int, retrieve(conf, "Architecture", "modes2"))
activation = retrieve(conf, "Architecture", "activation")
width = parse(Int, retrieve(conf, "Architecture", "channel_width"))
batch_size = parse(Int, retrieve(conf, "DataLoader", "batch_size"))

# Activation mapping
act_fcn = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish,
    "gelu" => NNlib.gelu
)[activation]

struct SpectralConv2d
    w1
    w2
    in_channels::Int64
    out_channels::Int64
end

function SpectralConv2d(in_channels::Int, out_channels::Int)
    scale = 1 / (in_channels * out_channels)
    weights1 = scale * randn(ComplexF32, modes1, modes2, in_channels, out_channels)
    weights2 = scale * randn(ComplexF32, modes1, modes2, in_channels, out_channels)
    return SpectralConv2d(weights1, weights2, in_channels, out_channels)
end

function compl_mul2d(input, weights, in_channels, out_channels)
    # Multiply the input with the weights "xyib,xyio->xyob"
    return CUDA.@allowscalar @tullio out[x, y, o, b] := input[x, y, i, b] * weights[x, y, i, o]
end

function (m::SpectralConv2d)(x)

    # Fourier transform using cuFFT
    x_FT = rfft(x, [1, 2]) 

    # Multiply relevant Fourier modes
    out_FT_1 = compl_mul2d(x_FT[1:modes1, 1:modes2, :, :], m.w1, m.in_channels, m.out_channels)
    out_FT_2 = compl_mul2d(x_FT[end-modes1+1:end, end-modes2+1:end, :, :], m.w2, m.in_channels, m.out_channels)
    
    out_FT = cat(out_FT_1, out_FT_2, dims=1)

    # Inverse fourier transform
    return irfft(out_FT, [1, 2])
end

struct MLP
    conv1
    conv2
    phi
end

function MLP(in_channels::Int64, out_channels::Int64, hidden_channels::Int64)
    mlp1 = Conv((1, 1), in_channels => hidden_channels)
    mlp2 = Conv((1, 1), hidden_channels => out_channels)
    return MLP(mlp1, mlp2, act_fcn)
end

function (m::MLP)(x)
    x = m.conv1(x)
    x = m.phi(x)
    return m.conv2(x)
end

Flux.@layer SpectralConv2d
Flux.@layer MLP

end

# Test gradient computation
using Flux
using .FNO_layers: SpectralConv2d, MLP

# Test the gradient computation for spect_conv

# Set up the model
model = SpectralConv2d(94, 94) |> gpu

# Set up the input
x = randn(32, 32, 94, 96) |> gpu


function loss(model, x)
    return sum(model(x))
end

# Compute the gradient of output w.r.t. params
loss_val, grads = Flux.withgradient(m -> loss(m, x), model)

println(grads[1])