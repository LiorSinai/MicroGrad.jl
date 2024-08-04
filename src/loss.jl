###
### Activation functions
###

""" 
    sigmoid(x) = 1 / (1 + exp(-x))

Sigmoid activation function (https://en.wikipedia.org/wiki/Sigmoid_function).
"""   
function sigmoid(x::Number)
    t = exp(-abs(x)) 
    ifelse(x ≥ 0, inv(1 + t), t / (1 + t)) # negative path is for numerical stability
end
sigmoid(x::AbstractArray) = sigmoid.(x)

function rrule(::typeof(sigmoid), x::Number)
    y = sigmoid(x)
    sigmoid_back(Δy) = (nothing, Δy * y * (1 - y))
    y, sigmoid_back
end

# Custom rules for AbstractArray because there is no generalised pullback for broadcasting
function rrule(::typeof(Broadcast.broadcasted), ::typeof(sigmoid), x::AbstractArray{<:Real})
    y = sigmoid.(x)
    sigmoid_back(Δy) = (nothing, nothing, Δy .* y .* (1 .- y)) 
    y, sigmoid_back
end

"""
    tanh_act(x) = tanh(x)

Hyperbolic tangent of `x`. Differs to the normal `tanh` in that it will broadcast over arrays instead of matrix operations.
"""
tanh_act(x::Number) = tanh(x)
tanh_act(x::AbstractArray) = tanh_act.(x)

function rrule(::typeof(tanh_act), x::Number)
    y = tanh(x)
    tanh_back(Δy) = (nothing, Δy * (1 - y^2))
    y, tanh_back
end

# Custom rules for AbstractArray because there is no generalised pullback for broadcasting
function rrule(::typeof(Broadcast.broadcasted), ::typeof(tanh_act), x::AbstractArray{<:Real})
    y = broadcast(tanh, x)
    tanh_back(Δy) = (nothing, nothing, Δy .* (1 .- y.^2))
    y, tanh_back
end

"""
    relu(x) = max(x, 0)

Rectified Linear Unit activation function (https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
"""
relu(x::Number) =  max(x, 0)
relu(x::AbstractArray) = max.(0, x)

function rrule(::typeof(relu), x::AbstractArray)
    y = relu(x)
    relu_back(Δy) = (nothing, ifelse.(x .> 0, Δy, 0))
    y, relu_back
end

"""
    logsoftmax(x) 

Computes the log of softmax in a more numerically stable way than directly taking `log.(softmax(xs))`.
"""
logsoftmax(x::AbstractArray) = x .- log.(sum(exp.(x), dims=1))

function rrule(::typeof(logsoftmax), x::AbstractArray)
    s = exp.(x)
    Σ = sum(s, dims=1)
    function logsoftmax_back(Δy)
        (nothing, Δy .- sum(Δy; dims=1) .* s ./ Σ)
    end
    x .- log.(Σ), logsoftmax_back
end

"""
    softmax(x)

Turn an input array into probabilities along the columns: each column will sum to 1 and all values will be in the range [0, 1].

The formula is: `exp.(x)/sum(exp.(x); dims=1)`
"""
softmax(x::AbstractArray) = exp.(x) ./ sum(exp.(x); dims=1)

function rrule(::typeof(softmax), x::AbstractArray)
    y = softmax(x)
    function softmax_back(Δy)
        tmp = Δy .* y
        (nothing, tmp .- y .* sum(tmp; dims=1))
    end
    y, softmax_back
end

###
### loss functions
###

mean(x::AbstractArray) = sum(x) / length(x)

# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/mapreduce.jl
function rrule(::typeof(mean), x::AbstractArray)
    y = mean(x)
    back_mean(Δy) = (nothing, _unsum(x, Δy) / length(x))
    y, back_mean
end

_unsum(x, Δy) = broadcast(last∘tuple, x, Δy) 

"""
    mse(ŷ, y)

Mean squared error: `mean((ŷ - y).^2)`.
"""
mse(ŷ::AbstractVecOrMat, y::AbstractVecOrMat) = mean(abs2.(ŷ - y))

function rrule(::typeof(mse), ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    Ω = mse(ŷ, y)
    function mse_back(Δy)
        c = 2 * (ŷ - y) / length(y) * Δy
        return nothing, c, -c # ∂self, ∂ŷ, ∂y
    end
    Ω, mse_back
end

"""
    cross_entropy(ŷ, y)

Cross entropy measures the number of bits (`log(e, x)`) required to encode an event from `ŷ` compared to the true distribution `y`.
A lower number of bits means more encoded knowledge.

Formula `mean(-sum(y .* log.(ŷ .+ 1e-6), dims=1))`
"""
function cross_entropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    mean(-sum(y .* log.(ŷ .+ 1e-6), dims=1))
end

function rrule(::typeof(cross_entropy), ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    Ω = cross_entropy(ŷ, y)
    function cross_entropy_back(Δ)
        size_l = size(Ω)
        n = length(size_l) > 1 ? sum(size(Ω)[2:end]) : 1
        ∂ŷ = -y ./ (ŷ .+ 1e-6) * Δ/n
        ∂y = -log.(ŷ)  * Δ/n
        return nothing, ∂ŷ , ∂y
    end
    Ω, cross_entropy_back
end

"""
    logit_cross_entropy(ŷ, y)

Logit cross entropy: the cross entropy after applying a `softmax` to `ŷ`.

Formula: `cross_entropy(softmax(ŷ), y) = mean(-sum(y .* logsoftmax(ŷ), dims=1))`.
"""
function logit_cross_entropy(ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    mean(-sum(y .* logsoftmax(ŷ), dims=1))
end

function rrule(::typeof(logit_cross_entropy),  ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    ls, logsoftmax_back = rrule(logsoftmax, ŷ)
    function logit_cross_entropy_back(Δ)
        size_ls = size(ls)
        n = length(size_ls) > 1 ? sum(size(ls)[2:end]) : 1
        ∂ŷ = -logsoftmax_back(y * Δ/n)[2]
        ∂y = Δ/n .* (-ls)
        return nothing, ∂ŷ , ∂y
    end
    mean(-sum(y .* ls, dims = 1)), logit_cross_entropy_back
end

"""
    hinge_loss(ŷ, y)

Also called max-margin loss.

Formula: `mean(relu(1 .- y .*  ŷ))`
"""
function hinge_loss(ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    mean(relu(1 .- y .*  ŷ))
end

function rrule(::typeof(hinge_loss),  ŷ::AbstractVecOrMat, y::AbstractVecOrMat)
    l = relu(1 .- y .* ŷ)
    function back_hinge_loss(Δ)
        n = length(l)
        ∂ŷ = -y .* ifelse.(l .> 0, Δ / n, 0)
        ∂y = ifelse.(l .> 0, -Δ/n .* ŷ, 0)
        return nothing, ∂ŷ , ∂y
    end
    mean(l), back_hinge_loss
end
