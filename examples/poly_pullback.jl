using Plots
using Printf
using LinearAlgebra

## model

struct Polynomial{V<:AbstractVector}
    weights::V
end

(m::Polynomial)(x) = evalpoly(x, m.weights)
(m::Polynomial)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)

function gradient_descent!(model, xs::AbstractVector, ys::AbstractVector; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100)
    losses = Float64[]
    for i in 1:max_iters
        # forward
        loss_iter, back = pullback(m -> mse(m(xs), ys), model)
        # reverse
        Δf, Δmodel = back(1.0)
        # update
        model.weights .-= learning_rate .* Δmodel.weights
        # history
        push!(losses, loss_iter)  
    end
    losses
end

function solve_poly_linear(order::Int, xs::AbstractVector, ys::AbstractVector, )
    @assert length(xs) == length(ys)
    n = length(xs)
    X = zeros(n, order + 1)
    for (i, x) in enumerate(xs)
        xpow = 1
        for j in 1:(size(X, 2))
            X[i, j] = xpow
            xpow *= x
        end
    end
    coeffs = pinv(X) * ys
    coeffs
end

target_weights = [15.0, -2.1, 13.9, 1.5]
xs = (rand(100) .- 0.5) .* 10
ys = map(x -> evalpoly(x, target_weights), xs)
ys .+= randn(length(ys)) .* abs.(ys)/maximum(ys) * 100
scatter(xs, ys)

weights = rand(4)
model = Polynomial(copy(weights)) # could also solve with a MLP and normalizing ys. See category.jl
coeffs_lin = solve_poly_linear(3, xs, ys)
history = gradient_descent!(model, xs, ys; learning_rate=1e-4, max_iters=200)

plot(1:length(history), history,
    title="History", label="", xlabel="steps", ylabel="loss")

y_model = model(xs)
ys_lin = map(x -> evalpoly(x, coeffs_lin), xs)
for (label, ys_est) in [("linear", ys_lin), ("descent", y_model)]
    e = mse(ys_est, ys)
    @printf "%-8s: %10.4f\n" label e
end

x_model = -5:0.01:5
y_lin = map(x -> evalpoly(x, coeffs_lin), x_model)
y_model = model(x_model)

scatter(xs, ys, label="data")
plot!(x_model, y_model, label="model")
plot!(x_model, y_lin, label="linear")
