using Plots; pythonplot()

include("./data_loader.jl")
using .loaders: get_darcy_loader

train_loader, test_loader = get_darcy_loader()
u_first = first(train_loader)[2]
print("u: ", size(u_first))

# Plotting the data
X, Y = [x for x in range(0, stop=1, length=32)], [y for y in range(0, stop=1, length=32)]

anim = @animate for u in eachslice(u_first, dims=1)
    contourf(X, Y, u, title="True Darcy Flow", aspect_ratio=:0.4, cbar=false, label="", color=:viridis)
end

# Save the animation to file
gif(anim, "figures/true_darcy_flow.gif", fps=5)


