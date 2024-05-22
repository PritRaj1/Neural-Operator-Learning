module FourierNO

export FNO

using NeuralOperators: FourierNeuralOperator
using ConfParser

conf = ConfParse("CNN_config.ini")
parse_conf!(conf)

width = parse(Int32, retrieve(conf, "Architecture", "channel_width"))
modes = parse(Int32, retrieve(conf, "Architecture", "modes"))
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

function FNO(in_channels::Int, out_channels::Int)
    channels = (in_channels, width, width, width, width, width, 2 * width, out_channels)
    return FourierNeuralOperator(ch = channels, modes = (modes,), σ = act_fcn)
end

end