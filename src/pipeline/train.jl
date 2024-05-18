module TRAINER

using Flux
using Optimisers

include("../utils.jl")

using .UTILS: log_loss

function train_model(m, train_loader, test_loader, opt_state, loss, num_epochs, model_name)

    @timed begin
        # Train the model
        for epoch in 1:num_epochs
            train_loss = 0.0
            test_loss = 0.0

            # Training
            for (x, y) in train_loader
                # Shape check
                train_loss, grad = Flux.withgradient(model -> loss(model, x, y), m)
                opt_state, m = Optimisers.update(opt_state, m, grad[1])
            end

            # Testing
            for (x, y) in test_loader
                test_loss += loss(m, x, y)
            end

            # Print progress
            train_loss /= length(train_loader.data)
            test_loss /= length(test_loader.data)
            println("Epoch $epoch: train_loss = $train_loss, test_loss = $test_loss")

            # Log the loss
            log_loss(epoch, train_loss, test_loss, model_name)
        end
    end

    return m
end
    
end
