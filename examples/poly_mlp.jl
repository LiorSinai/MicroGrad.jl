using Plots
using Printf
using LinearAlgebra

include("layers.jl")

function gradient_descent!(model, loss, xs::AbstractVecOrMat, ys::AbstractVecOrMat; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100)
    losses = Float64[]
    for i in 1:max_iters
        loss_iter, back = pullback(m -> loss(m(xs), ys), model)
        Δf, Δm = back(1.0)
        update!(parameters(model), Δm; learning_rate=learning_rate)
        push!(losses, loss_iter)  
    end
    losses
end

function update!(params::NamedTuple, grads::NamedTuple; options...)
    for key in keys(params)
        update!(params[key], grads[key]; options...)
    end
end

function update!(params::Tuple, grads::Tuple; options...)
    for (p, g) in zip(params, grads)
        update!(p, g; options...)
    end
end

function update!(params, grads; learning_rate::AbstractFloat=0.1)
    params .-= learning_rate .* grads # must broadcast to edit elements and not copies!
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

function normalize(x::AbstractVecOrMat)
    xmin, xmax= extrema(x)
    (x .- xmin) ./ (xmax - xmin)
end

target_weights = [15.0, -2.1, 13.9, 1.5]
xs = (rand(100) .- 0.5) .* 10
ys = map(x -> evalpoly(x, target_weights), xs)
ys .+= randn(length(ys)) .* abs.(ys)/maximum(ys) * 100
ys = normalize(ys)
scatter(xs, ys)

weights = rand(4)
X = reshape(xs, 1, :)
Y = reshape(ys, 1, :)

model = Chain(
    Dense(1 => 16, activation=tanh_act),
    Dense(16 => 16, activation=tanh_act),
    Dense(16 => 1, activation=sigmoid),
)
# Z, back = pullback(model, X)
# grads = back(1.0)

coeffs_lin = solve_poly_linear(3, xs, ys)
history = gradient_descent!(model, MicroGrad.mse, X, Y; learning_rate=0.1, max_iters=1_000)

plot(1:length(history), history,
    title="History", label="", xlabel="steps", ylabel="loss")

y_model = model(X) |> vec
ys_lin = map(x -> evalpoly(x, coeffs_lin), xs)
for (label, ys_est) in [("linear", ys_lin), ("descent", y_model)]
    e = mse(ys_est, ys)
    @printf "%-8s: %10.4f\n" label e
end

x_model = -5:0.01:5
y_lin = map(x -> evalpoly(x, coeffs_lin), x_model)
y_model = model(reshape(x_model, 1, :)) |> vec

scatter(xs, ys, label="data")
plot!(x_model, y_model, label="model")
plot!(x_model, y_lin, label="linear")
