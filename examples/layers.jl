using Random
using MicroGrad: relu
import Base.show

# https://github.com/FluxML/Flux.jl/blob/master/src/layers/basic.jl

struct Dense{M<:AbstractMatrix, B<:AbstractMatrix, F}
    weight::M
    bias::B
    activation::F
end

function (a::Dense)(x::AbstractVecOrMat)
    a.activation(a.weight * x .+ a.bias)
end

Dense((in, out)::Pair; activation=relu) = Dense(glorot_uniform(in, out), zeros(out, 1), activation)

parameters(a::Dense) = (;weight=a.weight, bias=a.bias)

function Base.show(io::IO, l::Dense)
    print(io, "Dense(", size(l.weight, 2), " => ", size(l.weight, 1))
    print(io, ", activation=", l.activation)
    print(io, ")")
end

struct Chain{T<:Tuple}
    layers::T
end
  
Chain(xs...) = Chain(xs)

(c::Chain)(x) = _apply_chain(c.layers, x)

@generated function _apply_chain(layers::Tuple{Vararg{Any,N}}, x) where {N}
  symbols = vcat(:x, [gensym() for _ in 1:N])
  calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
  Expr(:block, calls...)
end

parameters(c::Chain) = (;layers = map(parameters, c.layers))

function Base.show(io::IO, c::Chain)
    print(io, "Chain(")
    is_first = true
    for l in c.layers
        space = is_first ? "" : " "
        print(io, space, l, ",")
        is_first = false
    end
    print(io, ")")
end

"""
    glorot_uniform([rng], fan_in, fan_out) -> Array

Return an `Array{Float32}` of the given `size` containing random numbers drawn from a uniform
distribution on the interval ``[-x, x]``, where `x = sqrt(6 / (fan_in + fan_out))`.
"""
function glorot_uniform(rng::AbstractRNG, fan_in::Int, fan_out::Int)
    scale = sqrt(24 / (fan_in + fan_out))  # 0.5 * sqrt(24) = sqrt(1/4 * 24) = sqrt(6)
    (rand(rng, fan_out, fan_in) .- 0.5) .* scale
end

glorot_uniform(fan_in::Int, fan_out::Int) = glorot_uniform(Random.default_rng(), fan_in, fan_out)
