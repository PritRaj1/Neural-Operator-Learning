include("src/data_processing/data_loader.jl")
include("src/models/CNN.jl")
include("src/utils.jl")
include("src/pipeline/train.jl")

using .loaders: get_darcy_loader
using .TRAINER: train_model
using .UTILS: loss_fcn
using .ConvNN: CNN
using Flux
using Flux.Optimise: Adam, Descent
using ConfParser

conf = ConfParse("CNN_config.ini")
parse_conf!(conf)
batch_size = parse(Int, retrieve(conf, "DataLoader", "batch_size"))
num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))
optimizer_type = retrieve(conf, "Optimizer", "type")
LR = parse(Float32, retrieve(conf, "Optimizer", "learning_rate"))

train_loader, test_loader = get_darcy_loader(batch_size)
model = CNN(1, 1)

# Train the model
optimizer = Dict(
    "adam" => Flux.setup(Adam(LR), model),
    "sgd" => Flux.setup(Descent(LR), model)
)[optimizer_type]
loss = loss_fcn(2.0)
model = train_model(model, train_loader, test_loader, optimizer, loss, num_epochs)

# Save the model
Flux.save("trained_models/CNN_model.bson", model)