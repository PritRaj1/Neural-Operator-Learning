include("src/data_processing/data_loader.jl")
include("src/models/CNN.jl")
using Plots; pythonplot()
using Flux
using BSON: @load
using CUDA, KernelAbstractions
using .ConvNN: CNN
using .loaders: get_darcy_loader

train_loader, test_loader = get_darcy_loader(1)

MODEL_NAME = "CNN"

# Load the model
@load "trained_models/$MODEL_NAME.bson" model

# Plot the prediction
X, Y = [x for x in range(0, stop=1, length=32)], [y for y in range(0, stop=1, length=32)]#

anim = @animate for (a, u) in test_loader
    u_pred = model(cpu(a))
    u_pred = u_pred[:, :, 1, 1]
    contourf(X, Y, u_pred, title="$MODEL_NAME Prediction", cbar=false, color=:viridis)
end

# Save the animation to file
gif(anim, "figures/$MODEL_NAME" * "_prediction.gif", fps=5)





