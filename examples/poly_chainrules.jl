using Plots
using Printf
using LinearAlgebra
using StatsBase

# using Revise
using MicroGrad
import MicroGrad: rrule

## model

struct Polynomial{V<:AbstractVector}
    weights::V
end

(m::Polynomial)(x) = evalpoly(x, m.weights)
(m::Polynomial)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)
function rrule(m::Polynomial, x)
    y, back_poly = rrule(evalpoly, x, m.weights)
    function back_Polynomial(Δ)
        Δevalpoly, Δx, Δweights = back_poly(Δ)
        ((;weights=Δweights), Δx)
    end
    y, back_Polynomial
end
function rrule(m::Polynomial, x::AbstractVector)
    y, back_map = pullback(map, m, x)
    function back_Polynomial(Δ)
        Δmap, Δm, Δx = back_map(Δ)
        (Δm, Δx)
    end
    y, back_Polynomial
end

# https://towardsdatascience.com/polynomial-regression-gradient-descent-from-scratch-279db2936fe9

function gradient_descent_poly!(coeffs::AbstractVector, xs::AbstractVector, ys::AbstractVector; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100)
    history = Float64[]
    for i in 1:max_iters
        # forward
        ys_and_backs = map(x->rrule(evalpoly, x, coeffs), xs)
        ŷ = map(first, ys_and_backs)
        loss_iter, back_loss = rrule(mse, ŷ, ys)
        # reverse
        Δloss, Δŷ, Δy = back_loss(1.0)
        ∂f_and_∂x_zipped = map(((_, pb), δ) -> pb(δ), ys_and_backs, Δŷ)
        Δcoeffs_unzipped = map(Δ->Δ[3], ∂f_and_∂x_zipped) # Δ[i] = (Δevalpoly, Δx, Δcoeffs)
        Δcoeffs = reduce(+, Δcoeffs_unzipped)
        # update
        coeffs .-= learning_rate .* Δcoeffs
        # history
        push!(history, loss_iter)
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
        Δmse, Δŷ, Δy = back_loss(1.0)
        Δmodel, Δx = back_model(Δŷ)
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

function make_polynomial_label(coeffs::AbstractVector)
    parts = [@sprintf("%.1fx^%d", c, i) for (i, c) in enumerate(coeffs[2:end])]
    label = @sprintf "y=%.1f" coeffs[1]
    for x_str in parts
        if startswith(x_str, "-")
            label *= x_str
        else
            label *= "+" * x_str
        end
    end
    label
end

## Data
target_weights = [15.0, -2.1, 13.9, 1.5]
noise_factor = 0.2
xs = (rand(100) .- 0.5) .* 10
ys = map(x -> evalpoly(x, target_weights), xs)
scale_factor = mean(abs.(ys))
ys .+= randn(length(ys)) * scale_factor * noise_factor

scatter(xs, ys, label="", size=(900, 600), tickfontsize=14)
label = make_polynomial_label(target_weights)
x_target = -5:0.01:5
y_target = map(x -> evalpoly(x, target_weights), x_target)
plot!(x_target, y_target, label=label, linewidth=2, legendfontsize=14)

## Models

weights = rand(4)
model = Polynomial(copy(weights))
coeffs_lin = solve_poly_linear(3, xs, ys)
history1 = gradient_descent_poly!(weights, xs, ys; learning_rate=1e-5, max_iters=10_000)
history2 = gradient_descent!(model, xs, ys; learning_rate=1e-5, max_iters=10_000)

## Compare

ys1 = map(x -> evalpoly(x, weights), xs)
ys2 = model(xs)
ys_lin = map(x -> evalpoly(x, coeffs_lin), xs)
for (label, ys_est) in [("linear", ys_lin), ("descent poly", ys1), ("descent model", ys2)]
    e = mse(ys, ys_est)
    @printf "%-14s: %.4f\n" label e
end

y_lin = map(x -> evalpoly(x, coeffs_lin), x_target)
y_model1 = map(x -> evalpoly(x, weights), x_target)
y_model2 = model(x_target)

scatter(xs, ys, label="")
plot!(x_target, y_model1, label="gradient_descent_poly!")
plot!(x_target, y_model2, label="gradient_descent!")
plot!(x_target, y_lin, label="linear")
