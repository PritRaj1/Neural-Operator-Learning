include("src/data_processing/data_loader.jl")
include("src/models/CNN.jl")
include("src/utils.jl")
include("src/pipeline/train.jl")

using .loaders: get_darcy_loader
using .TRAINER: train_model
using .UTILS: LpLoss
using .ConvNN: CNN
using Flux, Optimisers
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
    "adam" => Flux.setup(Adam(), model),
    "sgd" => Flux.setup(SGD(LR), model)
)[optimizer_type]
loss_fcn = LpLoss(2.0)
model = train_model(model, train_loader, test_loader, optimizer, loss_fcn, num_epochs)

# Save the model
Flux.save("trained_models/CNN_model.bson", model)