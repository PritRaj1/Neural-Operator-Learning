module Transform_Layers

export encoder_layers, decoder_layers

using NNlib: softmax, batched_mul, batched_transpose
using Flux
using Flux: Chain, BatchNorm, LayerNorm, Dense, Dropout
using ConfParser
using CUDA, KernelAbstractions
using Tullio

conf = ConfParse("Transformer_config.ini")
parse_conf!(conf)

d_model = parse(Int, retrieve(conf, "Architecture", "d_model"))
nhead = parse(Int, retrieve(conf, "Architecture", "nhead"))
dim_feedforward = parse(Int, retrieve(conf, "Architecture", "dim_feedforward"))
max_len = parse(Int, retrieve(conf, "Architecture", "max_len"))
dropout = parse(Float32, retrieve(conf, "Architecture", "dropout"))
d_k = d_model ÷ nhead
query_mul = Float32.([d_k ^ (-0.5)]) |> gpu
sqrt_d_model = Float32.([sqrt(d_model)]) |> gpu

function scaled_dot_product_attention(query, key, value)
    key_T = batched_transpose(key)
    scores = batched_mul(query, key_T)
    scores = scores ./ sqrt_d_model
    p_attn = softmax(scores, dims=1)
    return batched_mul(p_attn, value)
end

function multi_head_attention(query, key, value)
    query = query .* query_mul
    return scaled_dot_product_attention(query, key, value)
end

function self_attention(x)
    return multi_head_attention(x, x, x)
end

struct encoder_layer
    self_attn
    feed_forward
    norm1
    norm2
end

function encoder_layers()
    feed_forward = Chain(
        Dense(d_model, dim_feedforward),
        Dropout(dropout),
        Dense(dim_feedforward, d_model),
        Dropout(dropout)
    ) 
    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    encoder_layer(self_attention, feed_forward, norm1, norm2)
end

function (l::encoder_layer)(x)
    x = l.norm1(x + l.self_attn(x))
    return l.norm2(x + l.feed_forward(x))
end

struct decoder_layer
    self_attn
    mh_attn
    feed_forward
    norm1
    norm2
    norm3
end

function decoder_layers()
    feed_forward = Chain(
        Dense(d_model, dim_feedforward),
        Dropout(dropout),
        Dense(dim_feedforward, d_model),
        Dropout(dropout)
    ) 
    norm1 = LayerNorm(d_model)
    norm2 = LayerNorm(d_model)
    norm3 = LayerNorm(d_model)
    decoder_layer(self_attention, multi_head_attention, feed_forward, norm1, norm2, norm3)
end

function (l::decoder_layer)((x, memory))
    x = batched_transpose(x)
    memory = batched_transpose(memory)
    x = l.norm1(batched_transpose(x + l.self_attn(x)))
    x = batched_transpose(x)
    x = l.norm2(batched_transpose(x + l.mh_attn(x, memory, memory)))
    return l.norm3(x + l.feed_forward(x))
end

Flux.@functor encoder_layer
Flux.@functor decoder_layer

end

