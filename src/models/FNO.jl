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

conf = ConfParse("../../FNO_config.ini")
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

struct FNO
    input_layer
    hidden_layers
    output_layer
end

# Construct the FNO model
function FNO(in_channels::Int, out_channels::Int, num_blocks::Int)
    phi = act_fcn

    input_layer = Dense(3 => width, phi)
    hidden_blocks = Chain(
        FNO_hidden_block(width, width),
        FNO_hidden_block(width, width),
        FNO_hidden_block(width, width),
        FNO_hidden_block(width, width)
    )
    
    output_MLP =  MLP(width, 1, width * 4)
    return FNO(input_layer, hidden_blocks, output_MLP)
end

function (m::FNO)(x)
    x = get_grid(x)
    x = m.input_layer(x)
    x = permutedims(x, [2, 3, 1, 4])
    x = m.hidden_layers(x)
    x = m.output_layer(x)
    return x
end

Flux.@layer FNO

end

# Test gradient computation

using .FourierNO: FNO
using Flux

model = FNO(3, 1, 4)
x = randn(32, 32, 1, 96)

model(x)
grads = Flux.gradient(model -> sum(model(x)), model)

