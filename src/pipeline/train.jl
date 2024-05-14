module TRAINER

using Flux

include("../utils.jl")

using .UTILS: log_loss

function train_model(model, train_loader, test_loader, optimizer, loss_fn, num_epochs)

    @timed begin
        # Train the model
        for epoch in 1:num_epochs
            train_loss = 0.0
            test_loss = 0.0

            # Training
            for (x, y) in train_loader
                # Shape check
                println(size(x))
                gs = gradient(() -> loss_fn(model(x), y), Flux.params(model))
                Flux.update!(optimizer, Flux.params(model), gs)
                train_loss += loss_fn(model(x), y)
            end

            # Testing
            for (x, y) in test_loader
                test_loss += loss_fn(model(x), y)
            end

            # Print progress
            train_loss /= length(train_loader.data)
            test_loss /= length(test_loader.data)
            println("Epoch $epoch: train_loss = $train_loss, test_loss = $test_loss")

            # Log the loss
            log_loss(epoch, train_loss, test_loss)
        end
    end

    return model
end
    
end
