module FourierNO

export FNO

include("../utils.jl")
include("./FNO_block.jl")
include("./FNO_layers.jl")

using .UTILS: get_grid
using .FNO_block: FNO_hidden_block
using .FNO_layers: MLP

using Flux
using Flux: Conv, Dense
using ConfParser
using NNlib

conf = ConfParse("FNO_config.ini")
parse_conf!(conf)

width = parse(Int, retrieve(conf, "Architecture", "channel_width"))
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

function permute(x)
    return permutedims(x, [2, 3, 1, 4])
end

struct input_block
    layers
end

# Construct the input block
function input_block()
    return input_block(Chain(
        get_grid,
        Dense(3, width),
        act_fcn
    ))
end

function (m::input_block)(x)
    return m.layers(x)
end

struct FNO
    layers
end

# Construct the FNO model
function FNO(in_channels::Int, out_channels::Int, num_blocks::Int)
    phi = act_fcn

    input_block = Chain(
        get_grid,
        Dense(3, width),
        permute,
        phi
    )

    hidden_blocks = [FNO_hidden_block(width, width) for i in 1:num_blocks]
    output_block = MLP(width, 1, width * 4)
    
    return FNO(Chain(input_block, hidden_blocks..., output_block))
end

function (m::FNO)(x)
    return m.layers(x)
end

Flux.@functor FNO

end
