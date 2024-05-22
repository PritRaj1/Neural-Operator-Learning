module FourierNeuralOperator

include("./layers.jl")

using Flux
using Flux: @functor, @layer
using Flux: rfft, irfft, Conv, gelu
using Layers: SpectralConv2d, MLP, compl_mul2d

struct FNO
    modes1::Int
    modes2::Int
    width::Int
    p::Dense
    conv0::SpectralConv2d
    conv1::SpectralConv2d
    conv2::SpectralConv2d
    conv3::SpectralConv2d
    mlp0::MLP
    mlp1::MLP
    mlp2::MLP
    mlp3::MLP
    w0::Conv
    w1::Conv
    w2::Conv
    w3::Conv
    act0::typeof(gelu)
    act1::typeof(gelu)
    act2::typeof(gelu)
    act3::typeof(gelu)
    q::MLP
end

function FNO(modes1, modes2, width)
    p = Dense(3, width)
    conv0 = SpectralConv2d(width, width, modes1, modes2)
    conv1 = SpectralConv2d(width, width, modes1, modes2)
    conv2 = SpectralConv2d(width, width, modes1, modes2)
    conv3 = SpectralConv2d(width, width, modes1, modes2)
    mlp0 = MLP(width, width, width)
    mlp1 = MLP(width, width, width)
    mlp2 = MLP(width, width, width)
    mlp3 = MLP(width, width, width)
    w0 = Conv((1, 1), width => width)
    w1 = Conv((1, 1), width => width)
    w2 = Conv((1, 1), width => width)
    w3 = Conv((1, 1), width => width)
    act0 = gelu
    act1 = gelu
    act2 = gelu
    act3 = gelu
    q = MLP(width, 1, width * 4)
    return FNO(modes1, modes2, width, p, conv0, conv1, conv2, conv3, mlp0, mlp1, mlp2, mlp3,
               w0, w1, w2, w3, act0, act1, act2, act3, q)
end

function (model::FNO)(x)
    grid = get_grid(size(x))
    x = cat(x, grid, dims=4)
    x = model.p(x)
    x = permutedims(x, (1, 4, 2, 3))

    x1 = model.conv0(x)
    x1 = model.mlp0(x1)
    x2 = model.w0(x)
    x = x1 + x2
    x = model.act0(x)

    x1 = model.conv1(x)
    x1 = model.mlp1(x1)
    x2 = model.w1(x)
    x = x1 + x2
    x = model.act1(x)

    x1 = model.conv2(x)
    x1 = model.mlp2(x1)
    x2 = model.w2(x)
    x = x1 + x2
    x = model.act2(x)

    x1 = model.conv3(x)
    x1 = model.mlp3(x1)
    x2 = model.w3(x)
    x = x1 + x2

    x = model.q(x)
    return dropdims(x, dims=2)
end

Flux.@functor FNO

end

# Test the FNO model
using FourierNeuralOperator

model = FourierNeuralOperator.FNO(8, 8, 64)
x = randn(64, 64, 2)
y = model(x)
println(size(y))