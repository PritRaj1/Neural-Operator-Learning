module TRAINER

using Flux
using Optimisers

include("../utils.jl")

using .UTILS: log_loss

LR = parse(Float32, get(ENV, "LR", "0.01"))
step = parse(Int, get(ENV, "step", "20"))
decay = parse(Float32, get(ENV, "decay", "0.8"))
min_LR = parse(Float32, get(ENV, "min_LR", "1e-4"))

# Step LR scheduler 
function step_decay(epoch)
    return max(LR * decay^(epoch // step), min_LR)
end

function train_model(m, train_loader, test_loader, opt_state, loss, num_epochs, model_name)

    @timed begin

        # Train the model
        for epoch in 1:num_epochs
            train_loss = 0.0
            test_loss = 0.0

            # Training
            for (x, y) in train_loader
                # Shape check
                loss_val, grad = Flux.withgradient(model -> loss(model, x, y), m)
                opt_state, m = Optimisers.update(opt_state, m, grad[1])
                train_loss += loss_val
            end

            # Testing
            for (x, y) in test_loader
                test_loss += loss(m, x, y)
            end

            # Update learning rate
            Optimisers.adjust!(opt_state, step_decay(epoch))

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
