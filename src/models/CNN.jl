module ConvNN

export CNN

using Flux, NNlib
using ConfParser

conf = ConfParse("CNN_config.ini")
parse_conf!(conf)

hidden_dim = parse(Int64, retrieve(conf, "Architecture", "hidden_dim"))
noise_injection = parse(Float64, retrieve(conf, "Architecture", "noise_injection"))
activation = retrieve(conf, "Architecture", "activation")

struct CNN
    encoder 
    decoder
end

# Activation mapping
act_fcn = Dict(
    "relu" => NNlib.relu,
    "leakyrelu" => NNlib.leakyrelu,
    "tanh" => NNlib.hardtanh,
    "sigmoid" => NNlib.hardsigmoid,
    "swish" => NNlib.hardswish
)[activation]

# Construct the CNN model
function CNN(in_channels::Int, out_channels::Int)
    phi = act_fcn
    noise_fcn = x -> x .+ noise_injection * randn(size(x))

    encoder = Chain(
        Conv((3, 3), in_channels => 2 * hidden_dim, phi; pad=1),
        noise_fcn,
        Conv((3, 3), 2 * hidden_dim => 4 * hidden_dim, phi; pad=1),
        noise_fcn,
        Conv((3, 3), 32 => 8 * hidden_dim, phi; pad=1),
        noise_fcn,
    )

    decoder = Chain(
        ConvTranspose((3, 3), 8 * hidden_dim => 4 * hidden_dim, phi; pad=1),
        noise_fcn,
        ConvTranspose((3, 3), 4 * hidden_dim => 2 * hidden_dim, phi; pad=1),
        noise_fcn,
        ConvTranspose((3, 3), 2 * hidden_dim => hidden_dim, phi; pad=1),
        noise_fcn,
        ConvTranspose((3, 3), hidden_dim => out_channels, phi; pad=1)
    )

    return CNN(encoder, decoder)
end

function (m::CNN)(x)
    return m.decoder(m.encoder(x))
end
    
end