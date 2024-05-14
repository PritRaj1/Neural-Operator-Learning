module loaders

include("./data_reader.jl")
include("../utils.jl")
using .MATFileLoader: load_darcy_data
using .UTILS: UnitGaussianNormaliser, encode, decode
using Flux

function get_darcy_loader(batch_size=32)
    a_train, u_train, a_test, u_test = load_darcy_data()

    a_normaliser = UnitGaussianNormaliser(a_train)
    u_normaliser = UnitGaussianNormaliser(u_train)

    # Normalise
    a_train = encode(a_normaliser, a_train)
    u_train = encode(u_normaliser, u_train)
    a_test = encode(a_normaliser, a_test)
    u_test = encode(u_normaliser, u_test)
    
    train_loader = Flux.DataLoader((a_train, u_train), batchsize=batch_size, shuffle=true)
    test_loader = Flux.DataLoader((a_test, u_test), batchsize=batch_size, shuffle=true)

    return train_loader, test_loader

end
end

