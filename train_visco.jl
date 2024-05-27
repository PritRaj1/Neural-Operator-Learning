using CUDA, KernelAbstractions
using Flux
using Optimisers
using ConfParser
using BSON: @save

MODEL_NAME = "Transformer"

# Parse config
conf = ConfParse(MODEL_NAME * "_config.ini")
parse_conf!(conf)

### Hyperparameters ###
batch_size = parse(Int, retrieve(conf, "DataLoader", "batch_size"))
num_epochs = parse(Int, retrieve(conf, "Pipeline", "num_epochs"))

optimizer_type = retrieve(conf, "Optimizer", "type")
LR = parse(Float32, retrieve(conf, "Optimizer", "learning_rate"))
min_LR = retrieve(conf, "Optimizer", "min_lr")
step = retrieve(conf, "Optimizer", "step_rate")
decay = retrieve(conf, "Optimizer", "gamma")

loss_norm_p = retrieve(conf, "Loss", "p")

# Set environment before loading files
ENV["p"] = loss_norm_p
ENV["step"] = step
ENV["decay"] = decay
ENV["LR"] = string(LR)
ENV["min_LR"] = min_LR

include("src/data_processing/data_loader.jl")
include("src/models/RNO.jl")
include("src/models/Transformer.jl")
include("src/utils.jl")
include("src/pipeline/train.jl")

using .loaders: get_visco_loader
using .TRAINER: train_model
using .UTILS: sequence_loss_fcn
using .RecurrentNO: createRNO
using .TransformerModel: createTransformer

train_loader, test_loader = get_visco_loader(batch_size)

in_size = size(first(train_loader)[2], 1)

model = Dict(
    "RNO" => gpu(createRNO(1, 1, in_size)),
    "Transformer" => gpu(createTransformer(in_size))
)[MODEL_NAME]

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

model = train_model(model, train_loader, test_loader, opt_state, sequence_loss_fcn, num_epochs, MODEL_NAME)

# Save the model
model = model |> cpu
@save "trained_models/$MODEL_NAME.bson" model