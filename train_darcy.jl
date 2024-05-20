include("src/data_processing/data_loader.jl")
include("src/models/CNN.jl")
include("src/utils.jl")
include("src/pipeline/train.jl")

using .loaders: get_darcy_loader
using .TRAINER: train_model
using .UTILS: loss_fcn
using .ConvNN: CNN
using Flux
using Optimisers
using ConfParser
using CUDA
using BSON: @save

MODEL_NAME = "CNN"

# Parse config
conf = ConfParse(MODEL_NAME * "_config.ini")
parse_conf!(conf)

### Hyperparameters ###
batch_size = parse(Int, retrieve(conf, "DataLoader", "batch_size"))
num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))
optimizer_type = retrieve(conf, "Optimizer", "type")
loss_norm_p = parse(Float32, retrieve(conf, "Loss", "p"))
LR = parse(Float32, retrieve(conf, "Optimizer", "learning_rate"))
ENV["p"] = loss_norm_p

get_model = Dict(
    "CNN" => CNN
)[MODEL_NAME]

train_loader, test_loader = get_darcy_loader(batch_size)
model = gpu(get_model(1, 1))

# Create logs directory if it doesn't exist
if !isdir("logs")
    mkdir("logs")
end

# Create new log file
open("logs/$MODEL_NAME.csv", "w") do file
    write(file, "epoch,train_loss,test_loss\n")
end

# Create trained_models directory if it doesn't exist
if !isdir("trained_models")
    mkdir("trained_models")
end 

# Train the model
opt_state = Dict(
    "adam" => Optimisers.setup(Optimisers.Adam(LR), model),
    "sgd" => Optimisers.setup(Optimisers.Descent(LR), model)
)[optimizer_type]

model = train_model(model, train_loader, test_loader, opt_state, loss_fcn, num_epochs, MODEL_NAME)

# Save the model
model = model |> cpu
@save "trained_models/$MODEL_NAME.bson" model