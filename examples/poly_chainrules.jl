using Plots
using Printf
using LinearAlgebra

# using Revise
using MicroGrad
import MicroGrad: rrule

## model

struct Polynomial{V<:AbstractVector}
    weights::V
end

(m::Polynomial)(x) = evalpoly(x, m.weights)
(m::Polynomial)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)
rrule(m::Polynomial, x) = rrule(evalpoly, x, m.weights)
rrule(m::Polynomial, x::AbstractVector) = pullback(map, m, x)

# https://towardsdatascience.com/polynomial-regression-gradient-descent-from-scratch-279db2936fe9

function gradient_descent_poly!(coeffs::AbstractVector, xs::AbstractVector, ys::AbstractVector; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100)
    history = Float64[]
    for i in 1:max_iters
        loss_total = 0.0
        for (x, y) in zip(xs, ys)
            y_est, back = rrule(evalpoly, x, coeffs)
            Δf, Δx, Δp = back(1.0)
            error = (y - y_est)^2
            Δerror = -2 * (y - y_est)
            loss_total += error
            coeffs .-= learning_rate .* Δp * Δerror
        end
        # history
        mean_loss = loss_total / length(xs)
        push!(history, mean_loss)
    end
    history
end

function gradient_descent!(model, xs::AbstractVector, ys::AbstractVector; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100)
    losses = Float64[]
    for i in 1:max_iters
        # forward
        ŷ, back_model = rrule(model, xs)
        loss_iter, back_loss = rrule(mse, ŷ, ys)
        # reverse
        Δf, Δŷ, Δy = back_loss(1.0)
        Δmap, Δf, Δx, Δweights = back_model(Δŷ)
        # update
        cum_grad = reduce(+, Δweights)
        model.weights .-= learning_rate .* cum_grad
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
model = Polynomial(copy(weights))
coeffs_lin = solve_poly_linear(3, xs, ys)
history1 = gradient_descent_poly!(weights, xs, ys; learning_rate=1e-5, max_iters=10_000)
history2 = gradient_descent!(model, xs, ys; learning_rate=1e-5, max_iters=10_000)

ys1 = map(x -> evalpoly(x, weights), xs)
ys2 = model(xs)
ys_lin = map(x -> evalpoly(x, coeffs_lin), xs)
for (label, ys_est) in [("linear", ys_lin), ("descent poly", ys1), ("descent model", ys2)]
    e = mse(ys, ys_est)
    @printf "%-14s: %.4f\n" label e
end

x_model = -5:0.01:5
y_lin = map(x -> evalpoly(x, coeffs_lin), x_model)
y_model1 = map(x -> evalpoly(x, weights), x_model)
y_model2 = model(x_model)

scatter(xs, ys)
plot!(x_model, y_model1)
plot!(x_model, y_model2)
plot!(x_model, y_lin)
