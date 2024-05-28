module TransformerModel

export createTransformer

include("./transformer_layers.jl")

using Flux
using Flux: Chain, Dense
using ConfParser
using CUDA, KernelAbstractions
using .Transform_Layers: encoder_layers, decoder_layers

conf = ConfParse("Transformer_config.ini")
parse_conf!(conf)

d_model = parse(Int, retrieve(conf, "Architecture", "d_model"))
num_encoder_layers = parse(Int, retrieve(conf, "Architecture", "num_encoder_layers"))
num_decoder_layers = parse(Int, retrieve(conf, "Architecture", "num_decoder_layers"))
max_len = parse(Int, retrieve(conf, "Architecture", "max_len"))
dropout = parse(Float32, retrieve(conf, "Architecture", "dropout"))
num_history = parse(Int, retrieve(conf, "Architecture", "num_history"))

struct PositionEncoding
    pe_vector
end

function PositionalEncoding()
    pe_vector = zeros(Float32, d_model, max_len)
    position = Float32.(range(1, max_len))
    div_term = exp.(Float32.(-log(10000.0) .* range(1, d_model, step=2) ./ d_model))
    div_term = reshape(div_term, 1, floor(Int, d_model/2))
    pe_vector[1:2:end, :] = transpose(sin.(position .* div_term))
    pe_vector[2:2:end, :] = transpose(cos.(position .* div_term))
    pe_vector = Float32.(reshape(pe_vector, d_model, max_len, 1)) |> gpu
    PositionEncoding(pe_vector)
end

function (pe::PositionEncoding)(x)
    x = reshape(x, 1, size(x, 1), size(x, 2)) 
    return x .+ pe.pe_vector[:, 1:size(x, 2), :]
end

struct Transformer
    position_encoding::PositionEncoding
    encoder
    decoder
    output_layer
end

function createTransformer(tgt_size)
    position_encoding = PositionalEncoding()
    encoder = [encoder_layers() for _ in 1:num_encoder_layers]
    decoder = [decoder_layers() for _ in 1:num_decoder_layers]
    output_layer = Dense(d_model, tgt_size)
    Transformer(position_encoding, Chain(encoder...), Chain(decoder...), output_layer)
end

function (m::Transformer)(src, tgt)
    
    # Predict from first num_history time steps
    y = reshape(tgt[1:num_history, :], num_history, size(tgt)[end])

    src = m.position_encoding(src)
    tgt = m.position_encoding(y)

    memory = m.encoder(src)
    output = m.decoder((tgt, memory))
    output = m.output_layer(output)
    output = output[:, end, :]
    
    return reshape(output, size(output, 1), size(output)[end])

end

Flux.@functor Transformer

end