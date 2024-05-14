using Plots; pythonplot()

include("./data_loader.jl")
using .loaders: get_darcy_loader

train_loader, test_loader = get_darcy_loader(1)
u_first = first(train_loader)[2]
print("u: ", size(u_first))

# Plotting the data
X, Y = [x for x in range(0, stop=1, length=32)], [y for y in range(0, stop=1, length=32)]

anim = @animate for (a, u) in test_loader
    u = u[:, :, 1, 1]
    contourf(X, Y, u, title="True Darcy Flow", cbar=false, color=:viridis)
end

# Save the animation to file
gif(anim, "figures/true_darcy_flow.gif", fps=5)


