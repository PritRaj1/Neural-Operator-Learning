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
            model = Flux.train!(loss_fn, model, train_loader, optimizer) do batch_size, model, loss
                train_loss += batch_size * loss
            end

            # Testing
            model = Flux.test!(model, test_loader, loss_fn) do batch_size, model, loss
                test_loss += batch_size * loss
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