using Plots; pythonplot()
using CUDA
using Flux

include("./data_loader.jl")
using .loaders: get_darcy_loader, get_visco_loader

train_loader, test_loader = get_darcy_loader(1)
u_first = first(train_loader)[2]
print("u: ", size(u_first))

# Plotting the data
X, Y = [x for x in range(0, stop=1, length=32)], [y for y in range(0, stop=1, length=32)]

anim = @animate for (a, u) in test_loader
    u = u |> cpu
    u = u[:, :, 1, 1]
    contourf(X, Y, u, title="True Darcy Flow", cbar=false, color=:viridis)
end

# Save the animation to file
gif(anim, "figures/true_darcy_flow.gif", fps=5)

train_loader, test_loader = get_visco_loader(1)

epsi_first, sigma_first = first(test_loader) |> cpu
num_samples = size(epsi_first, 1)

anim = @animate for i in 1:num_samples
    epsi = epsi_first[1:i,1]
    sigma = sigma_first[1:i,1]  
    plot(epsi, sigma, title="True Viscoplastic Data", xlabel="Strain", ylabel="Stress", color=:blue, label="Test Sample 1")
    xlims!(0, 1)
    ylims!(0, 1)
end

# Save the animation to file
gif(anim, "figures/true_test_visco_data.gif", fps=10)

epsi_first, sigma_first = first(train_loader) |> cpu
num_samples = size(epsi_first, 1)

anim = @animate for i in 1:num_samples
    epsi = epsi_first[1:i,1]
    sigma = sigma_first[1:i,1]  
    plot(epsi, sigma, title="True Viscoplastic Data", xlabel="Strain", ylabel="Stress", color=:red, label="Train Sample 1")
    xlims!(0, 1)
    ylims!(0, 1)
end

# Save the animation to file
gif(anim, "figures/true_train_visco_data.gif", fps=10)