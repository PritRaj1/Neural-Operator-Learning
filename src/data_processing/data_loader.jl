module loaders

include("./data_reader.jl")
include("../utils.jl")
using .MATFileLoader: load_darcy_data, load_visco_data
using .UTILS: UnitGaussianNormaliser, unit_encode, MinMaxNormaliser, minmax_encode
using Flux
using CUDA

function get_darcy_loader(batch_size=32)
    a_train, u_train, a_test, u_test = load_darcy_data()

    a_normaliser = UnitGaussianNormaliser(a_train)
    u_normaliser = UnitGaussianNormaliser(u_train)

    # Normalise
    a_train = unit_encode(a_normaliser, a_train)
    u_train = unit_encode(u_normaliser, u_train)
    a_test = unit_encode(a_normaliser, a_test)
    u_test = unit_encode(u_normaliser, u_test)

    a_train = reshape(a_train, size(a_train)..., 1)
    u_train = reshape(u_train, size(u_train)..., 1)
    a_test = reshape(a_test, size(a_test)..., 1)
    u_test = reshape(u_test, size(u_test)..., 1)

    a_train = permutedims(a_train, [2, 3, 4, 1])
    u_train = permutedims(u_train, [2, 3, 4, 1])
    a_test = permutedims(a_test, [2, 3, 4, 1])
    u_test = permutedims(u_test, [2, 3, 4, 1])
    
    train_loader = Flux.DataLoader((a_train, u_train) |> gpu, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((a_test, u_test) |> gpu, batchsize=batch_size, shuffle=false)

    return train_loader, test_loader
end

function get_visco_loader(batch_size=32)
    epsi_train, sigma_train, epsi_test, sigma_test = load_visco_data()

    epsi_normaliser = MinMaxNormaliser(epsi_train)
    sigma_normaliser = MinMaxNormaliser(sigma_train)

    # Down-sample the data to a coarser grid in time - reduces the training time
    s = 4
    epsi_train = epsi_train[:, 1:s:end]
    sigma_train = sigma_train[:, 1:s:end]
    epsi_test = epsi_test[:, 1:s:end]
    sigma_test = sigma_test[:, 1:s:end]

    # Normalise
    epsi_train = minmax_encode(epsi_normaliser, epsi_train)
    sigma_train = minmax_encode(sigma_normaliser, sigma_train)
    epsi_test = minmax_encode(epsi_normaliser, epsi_test)
    sigma_test = minmax_encode(sigma_normaliser, sigma_test)

    epsi_train = permutedims(epsi_train, [2, 1])
    sigma_train = permutedims(sigma_train, [2, 1])
    epsi_test = permutedims(epsi_test, [2, 1])
    sigma_test = permutedims(sigma_test, [2, 1])

    train_loader = Flux.DataLoader((epsi_train, sigma_train) |> gpu, batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((epsi_test, sigma_test) |> gpu, batchsize=batch_size, shuffle=false)

    return train_loader, test_loader
    
end
end

