# https://github.com/JuliaDiff/ChainRulesCore.jl/blob/main/src/rules.jl
"""
    rrule(f, xs...)

Returns the tuple: `(f(xs...), back)` where `back(∂y/∂f) = (∂f, ∂x₁, ∂x₂, ... )`. Here `∂x` is shorthand for `∂y/∂x`.

`back(Δ) = J'Δ` where `J=∂f/∂x` is the derivative/Jacobian (if inputs are arrays) and `Δ=∂y/∂f` is the incoming gradient.
Note the Jacobian does not need to be explicitly calculated; only the product needs to be.

If no method matching `rrule(f, xs...)` has been defined, then return nothing.
"""
rrule(::Any, ::Vararg{Any}) = nothing

# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/fastmath_able.jl
function rrule(::typeof(+), x::Number, y::Number)
    add_back(Δy) = (nothing, true * Δy, true * Δy) # ∂f, ∂x, ∂y
    x + y, add_back
end

function rrule(::typeof(-), x::Number, y::Number)
    minus_back(Δy) = (nothing, true * Δy, -1 * Δy) # ∂f, ∂x, ∂y
    x - y, minus_back
end

function rrule(::typeof(*), x::Number, y::Number)
    times_back(Δy) = (nothing, y * Δy, x * Δy) # ∂f, ∂x, ∂y
    x * y, times_back
end

function rrule(::typeof(/), x::Number, y::Number)
    z = x / y
    divide_back(Δy) = (nothing, 1 / y * Δy, -z/y * Δy) # ∂f, ∂x, ∂y
    z, divide_back
end

function rrule(::typeof(exp), x::Real)
    y = exp(x)
    y, Δy -> Δy * y
end

###
### Trigonometry
###

# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/fastmath_able.jl
function rrule(::typeof(sin), x::Number)
    s, c = sincos(x)
    sin_back(Δy) = (nothing, Δy * c) # ∂f, ∂x
    s, sin_back
end

function rrule(::typeof(cos), x::Number)
    s, c = sincos(x)
    cos_back(Δy) = (nothing, -Δy * s) # ∂f, ∂x
    c, cos_back
end

###
### Polynomial
###

# y = coeffs[1]+coeffs[2]*x^1 + ... + coeffs[n]*x^(n-1)
# ∂y/∂p[i] = x^(i-1)
# ∂y/∂x = 0 + coeffs[2] + 2*coeffs[3]*x + ... + (n-1)*coeffs[n]*x^(n-2)
# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/evalpoly.jl
function rrule(::typeof(evalpoly), x, coeffs::AbstractVector)
    y = evalpoly(x, coeffs)
    function evalpoly_back(Δy)
        xpow = one(x)
        dp = similar(coeffs, typeof(xpow * Δy))
        dx = zero(x)
        for i in eachindex(coeffs)
            dp[i] = Δy * xpow
            dx += (i-1) * coeffs[i] * xpow / x * Δy
            xpow *= x
        end
        return nothing, dx, dp
    end
    y, evalpoly_back
end

###
### Arrays
###

function rrule(::typeof(*), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    function times_back(ΔY)
        dA = ΔY * B'
        dB = A' * ΔY
        return (nothing, dA, dB)
    end
    A * B, times_back
end

function rrule(::typeof(+), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    function plus_back(ΔY)
        return (nothing, ΔY, ΔY)
    end
    A + B, plus_back
end

# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/broadcast.jl
# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/broadcast.jl
# Warning: Zygote mentions that broadcast AD is complex and can slow down performance if not done correctly.
# materialize(bc::Broadcasted) = copy(instantiate(bc))
# broadcast(f::Tf, As...) where {Tf} = materialize(broadcasted(f, As...))

function rrule(::typeof(copy), bc::Broadcast.Broadcasted)
    uncopy(Δ) = (nothing, Δ)
    return copy(bc), uncopy
end

function rrule(::typeof(Broadcast.instantiate), bc::Broadcast.Broadcasted)
    uninstantiate(Δ) = (nothing, Δ)
    return Broadcast.instantiate(bc), uninstantiate
end

function unbroadcast(x::AbstractArray, x̄)
    if length(x) == length(x̄)
        x̄
    else
      dims = ntuple(d -> size(x, d) == 1 ? d : ndims(x̄)+1, ndims(x̄))
      dx = sum(x̄; dims = dims)
      check_dims(size(x), size(dx))
      dx
    end
end

function check_dims(size_x, size_dx) # see ChainRulesCore.ProjectTo
    for (i, d) in enumerate(size_x)
        dd = i <= length(size_dx) ? size_dx[i] : 1 # broadcasted dim
        if d != dd 
            throw(DimensionMismatch("variable with size(x) == $size_x cannot have a gradient with size(dx) == $size_dx"))
        end
    end
end

function rrule(::typeof(Broadcast.broadcasted), ::typeof(+), A::AbstractVecOrMat{<:Real}, B::AbstractVecOrMat{<:Real})
    broadcast_back(Δy) = (nothing, nothing, unbroadcast(A, Δy), unbroadcast(B, Δy))
    broadcast(+, A, B), broadcast_back
end

# https://github.com/JuliaDiff/ChainRules.jl/blob/main/src/rulesets/Base/indexing.jl
function rrule(::typeof(getindex), x::AbstractArray, inds...)
    Δinds = map(Returns(nothing), inds)
    function getindex_pullback(Δy)
        Δx = zeros(eltype(x), size(x)...)
        Δx[inds...] += Δy
        (nothing, Δx, Δinds...)
    end
    return x[inds...], getindex_pullback
end

function rrule(::typeof(getindex), x::T, i::Integer) where {T<:Tuple}
    function getindex_back_1(Δy)
        dx = ntuple(j -> j == i ? Δy : nothing, length(x))
        return (nothing, (dx...,), nothing)
    end
    return x[i], getindex_back_1
end