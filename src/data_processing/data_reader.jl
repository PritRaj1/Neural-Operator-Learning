module MATFileLoader

export load_darcy_data

using MAT

function load_darcy_data()
    train_matfile = matread("DATA/2D_DarcyFlow/Darcy_2D_data_train.mat")
    test_matfile = matread("DATA/2D_DarcyFlow/Darcy_2D_data_test.mat")

    a_train = Float32.(train_matfile["a_field"])
    u_train = Float32.(train_matfile["u_field"])
    a_test = Float32.(test_matfile["a_field"])
    u_test = Float32.(test_matfile["u_field"])

    return a_train, u_train, a_test, u_test
end  

function load_visco_data()
    
    N_total = parse(Int, get(ENV, "N_total", "400"))
    N_train = parse(Int, get(ENV, "N_train", "300"))
    N_test = N_total - N_train

    matfile = matread("DATA/1D_Viscoplastic/viscodata_3mat.mat")

    epsi_field = Float32.(matfile["epsi_tol"])[1:N_total, :]
    sigma_field = Float32.(matfile["sigma_tol"])[1:N_total, :]

    # Split data
    epsi_train = epsi_field[1:N_train, :]
    sigma_train = sigma_field[1:N_train, :]
    epsi_test = epsi_field[N_train+1:end, :]
    sigma_test = sigma_field[N_train+1:end, :]

    return epsi_train, sigma_train, epsi_test, sigma_test
end
end