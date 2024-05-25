module FNO_block

export FNO_hidden_block

include("./FNO_layers.jl")

using Flux
using Flux: Conv, Chain
using .FNO_layers: SpectralConv2d, MLP

using ConfParser
using NNlib

conf = ConfParse("../../FNO_config.ini")
parse_conf!(conf)

width = parse(Int64, retrieve(conf, "Architecture", "channel_width"))
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

struct FNO_hidden_block
    spect_conv
    mlp
    conv
    phi
end

function FNO_hidden_block(in_channels::Int, out_channels::Int)
    spect_conv = SpectralConv2d(in_channels, out_channels)
    mlp = MLP(width, width, width)
    conv = Conv((1, 1), width => width)
    return FNO_hidden_block(spect_conv, mlp, conv, act_fcn)
end

function (m::FNO_hidden_block)(x)
    x2 = m.conv(x)
    println("x2: ", size(x2))
    x = m.spect_conv(x)
    println("x: ", size(x))
    x = m.mlp(x)
    println("x: ", size(x))
    y = x .+ x2
    println("y: ", size(y))
    y = m.act_fcn(y)
    return m.act_fcn(x .+ x2)
end

Flux.@layer FNO_hidden_block

end

# Test gradient computation
using CUDA 
using Flux
using .FNO_block: FNO_hidden_block

# Test gradient computation
model = FNO_hidden_block(94, 94) |> gpu

x = randn(32, 32, 94, 1) |> gpu

model(x)

loss = sum(model(x))

loss, grad = Flux.withgradient(Flux.params(model)) do
    sum(model(x))
end

println(grads)