module DataLoader

include("./data_read.jl")
using .MATFileLoader: load_darcy_data


function get_darcy_loader()
    a_train, u_train, a_test, u_test = load_darcy_data()
end
end
