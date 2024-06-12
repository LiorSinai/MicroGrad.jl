using Plots
using MicroGrad
using Printf

include("datasets.jl")
include("layers.jl")

function get_batch(X::AbstractVecOrMat, Y::AbstractVecOrMat, batch_size::Int)
    inds_start = ntuple(Returns(:), ndims(X) - 1)
    inds_batch = rand(1:size(X)[end], batch_size) 
    x = X[inds_start..., inds_batch]
    inds_start = ntuple(Returns(:), ndims(Y) - 1)
    y = Y[inds_start..., inds_batch]
    x, y
end

get_batch(X::AbstractVecOrMat, Y::AbstractVecOrMat, batch_size::Nothing) = (X, Y)

function gradient_descent!(model, loss, X::AbstractVecOrMat, Y::AbstractVecOrMat; 
    learning_rate::AbstractFloat=0.1, max_iters::Integer=100, batch_size::Union{Nothing, Int}=32)
    losses = Float64[]
    for i in 1:max_iters
        xb, yb = get_batch(X, Y, batch_size)
        loss_iter, back = pullback(model) do m
            result = m(xb)
            loss(result, yb)
        end 
        Δf, Δm = back(1.0)
        update!(parameters(model), Δm; learning_rate=learning_rate)
        push!(losses, loss_iter)  
        acc = accuracy(model, X, Y) * 100
        @printf "Step %4d - loss %.4f - accuracy %.4f%%\n" i loss_iter acc
    end
    losses
end

# function update!(model::Dense, grads; learning_rate=0.1)
#     model.bias .-= learning_rate * grads.bias
#     model.weight .-= learning_rate * grads.weight
# end

# function update!(model::Chain, grads; options...)
#     for (layer, g) in zip(model.layers, grads.layers)
#         update!(layer, g; options...)
#     end
# end

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

### Category

function onehot(y::AbstractVector, labels)
    # there are more efficient implementations using SparseArrays
    num_classes = maximum(labels)
    Y = zeros(num_classes, length(y))
    for (j, label) in enumerate(y)
        Y[label, j] += 1
    end
    Y
end

function onecold(Y::AbstractMatrix)
    labels = map(idx -> idx[1], argmax(Y, dims=1))
    vec(labels)
end

function normalize(x::AbstractVecOrMat)
    xmin, xmax= extrema(x)
    (x .- xmin) ./ (xmax - xmin)
end

###################################
### Bivariate normals
function generate_data(μ::Tuple, σ::Tuple, ρ::Float64, n::Int)
    z1 = randn(n)
    z2 = randn(n)
    xs = μ[1] .+ (σ[1] .* z1)
    ys = μ[2] .+ σ[2] .* (ρ * z1 .+ sqrt(1 - ρ*ρ) .* z2)
    permutedims(hcat(xs, ys))
end

μ1, σ1 = (-1.0, -1.0), (0.5, 0.5)
μ2, σ2 = (0.5, 0.5), (0.8, 0.5) # intentionally overlaps with the other two
μ3, σ3 = (1.5, 1.5), (0.1, 0.1)

n = 100
nlabels = 3
X1 = generate_data(μ1, σ1, 0.0, n)
X2 = generate_data(μ2, σ2, 0.0, n)
X3 = generate_data(μ3, σ3, 0.0, n)

label_markers = [:circle, :square, :diamond]
label_palette = palette(:default)
scatter(X1[1, :], X1[2, :]; aspectratio=:equal, marker=label_markers[1])
scatter!(X2[1, :], X2[2, :], marker=label_markers[2])
scatter!(X3[1, :], X3[2, :], marker=label_markers[3])

X = hcat(X1, X2, X3)
#X = normalize(X)
y = vcat(map(i -> fill(i, n), 1:3)...)
Y = onehot(y, 1:3)

# model = Dense(2 => nlabels, activation=tanh_act)
model = Chain(
    Dense(2 => 16, activation=relu),
    Dense(16 => 16, activation=relu),
    Dense(16 =>nlabels, activation=relu)
)

loss(ŷ, ys) = MicroGrad.logit_cross_entropy(ŷ, ys)
accuracy(model, X, Y) = MicroGrad.mean(onecold(model(X)) .== onecold(Y))

random_loss = -log(1/nlabels)
init_loss = loss(model(X), Y)

history = gradient_descent!(
    model, loss, X, Y; 
    learning_rate=0.05, max_iters=500, batch_size=64
);

plot(1:length(history), history, title="History", label="", xlabel="steps", ylabel="loss")

ŷ = onecold(model(X))

markers = vcat(map(s->fill(s, n), label_markers)...)
canvas = scatter(X[1, :], X[2, :], color=ŷ, label="", aspectratio=:equal, markershape=markers)
# plot dummy points offscreeen for the legend
canvas_xlims = xlims(canvas)
canvas_ylims = ylims(canvas)
scatter!([-1000], [-1000], xlims=canvas_xlims, ylims=canvas_ylims, label="1", marker=label_markers[1], color=label_palette[1])
scatter!([-1000], [-1000], xlims=canvas_xlims, ylims=canvas_ylims, label="2", marker=label_markers[2], color=label_palette[2])
scatter!([-1000], [-1000], xlims=canvas_xlims, ylims=canvas_ylims, label="3", marker=label_markers[3], color=label_palette[3])

is_correct = ifelse.(ŷ .== y, :green, :red)
scatter(X[1, :], X[2, :], 
    color=is_correct, label="",aspectratio=:equal, markershape=markers)

###################################
## Moons
n = 200
nlabels = 2
n_moons = floor(Int, n / 2)
X = make_moons(n; noise=0.1)
#X = normalize(X)
y = vcat(fill(1, n_moons)..., fill(2, n - n_moons)...)
Y = onehot(y, 1:nlabels)
scatter(X[1, :], X[2, :], color=y, label="")

model = Chain(
    Dense(2 => 16, activation=relu),
    Dense(16 => 16, activation=relu),
    Dense(16=>2, activation=relu)
)
# Z, back = pullback(model, X)
# Z, back = pullback(loss, model(X), Y)
# Z, back = pullback(m -> loss(m(X), Y), model)
# grads = back(1.0)

loss(ŷ, ys) = MicroGrad.logit_cross_entropy(ŷ, ys)
accuracy(model, X, Y) = MicroGrad.mean(onecold(model(X)) .== onecold(Y))

random_loss = -log(1/nlabels) # logit_cross_entropy
init_loss = loss(model(X), Y)

history = gradient_descent!(model, loss, X, Y
    ; learning_rate=0.5, max_iters=500, batch_size=nothing);

plot(1:length(history), history,
    title="History", label="", xlabel="steps", ylabel="loss")

ŷ = onecold(model(X))
final_loss = loss(model(X), Y)
scatter(X[1, :], X[2, :], color=ŷ, label="", aspectratio=:equal)

is_correct = ifelse.(ŷ .== y, :green, :red)
scatter(X[1, :], X[2, :], color=is_correct, label="", aspectratio=:equal)

## Decision boundary

xmin, xmax = extrema(X[1, :])
ymin, ymax = extrema(X[2, :])
h = 0.01
xrange = (xmin-0.1):h:(xmax+0.1)
yrange = (ymin-0.1):h:(ymax+0.1)

x_grid = xrange' .* ones(length(yrange))
y_grid = ones(length(xrange))' .* yrange
Z = similar(x_grid)
for idx in eachindex(x_grid)
    score = model([x_grid[idx], y_grid[idx]])
    #Z[idx] = softmax(score)[1]
    Z[idx] = argmax(vec(score))
end

canvas = heatmap(xrange, yrange, Z, size=(800, 500))
scatter!(
    X[1, :], X[2, :], color=y, label="", aspectratio=:equal,
    #markershape=markers,
    xlims = xlims(canvas),
    ylims = ylims(canvas),
)
