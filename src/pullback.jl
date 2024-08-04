# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/lib.jl

accum() = nothing
accum(x) = x
accum(x, y) = x === nothing ? y : y === nothing ? x : x + y
accum(x::Tuple, ys::Tuple...) = map(accum, x, ys...)
accum(x, y, zs...) = accum(accum(x, y), zs...)

@generated function accum(x::NamedTuple, y::NamedTuple)
    # assumes that y has no keys apart from those also in x
    fieldnames(y) ⊆ fieldnames(x) || throw(ArgumentError("$y keys must be a subset of $x keys"))
    grad(field) = field in fieldnames(y) ? :(y.$field) : :nothing
    Expr(:tuple, [:($f=accum(x.$f, $(grad(f)))) for f in fieldnames(x)]...)
end

"""
    pullback(f, xs...)

Returns the value of the function `f` and a back-propagator function which can be called to obtain a tuple containing the derivative
`∂f/∂x` for each argument `x`.

    y, back = pullback(f, xs...)
    ∇ = back(seed)

`back` must be called with a start value seed matching the output of `f(xs...)`.

The rules for `pullback` are:
  - If a dispatch rule is defined for the arguments then that takes priority.
  - If an `rrule` exists then `pullback` passes the arguments through to `rrule`.
  - Otherwise the code will be inspected and a `Pullback` struct returned with pullbacks for each function call.
"""
pullback

### 
### map
###

# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/array.jl
struct StaticGetter{i} end
(::StaticGetter{i})(v) where {i} = v[i]
(::StaticGetter{i})(::Nothing) where {i} = nothing

function _unzip(tuples, ::Val{N}) where {N}
  getters = ntuple(n -> StaticGetter{n}(), N)
  map(g -> map(g, tuples), getters)
end

function unzip(tuples)
  N = length(first(tuples))
  _unzip(tuples, Val(N))
end

function pullback(::typeof(map), f, xs)
    ys_and_backs = map((xs...) -> pullback(f, xs...), xs)
    ys = map(first, ys_and_backs)
    function map_pullback(Δ)
      # technically should apply f in reverse and reverse back afterwards in case f is stateful
      ∂f_and_∂x_zipped = map(((_, pb), δ) -> pb(δ), ys_and_backs, Δ)
      ∂f_and_∂x = unzip(∂f_and_∂x_zipped) 
      ∂f = reduce(accum, ∂f_and_∂x[1])
      ∂args = ∂f_and_∂x[2:end]
      return (nothing, ∂f, ∂args...)
    end
    ys, map_pullback
end

###
### getfield
###
# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/lib.jl

literal_getfield(x, ::Val{f}) where f = getfield(x, f)

@generated nt_nothing(x) = Expr(:tuple, [:($f=nothing) for f in fieldnames(x)]...)
@generated pair(::Val{k}, v, _=nothing) where k = :($k = v,)

function pullback(::typeof(literal_getfield), x, ::Val{f}) where f
  val = getfield(x, f)
  function literal_getfield_back(Δ)
    if isimmutable(x)
      dx = (; nt_nothing(x)..., pair(Val(f), Δ)...)
      (nothing, dx, nothing)
    else
      error("multable stucts not supported")
    end
  end
  val, literal_getfield_back
end

pullback(::typeof(getfield), x, field_name::Symbol) = pullback(literal_getfield, x, Val(field_name))
# getproperty by default calls getfield, but it can have custom results and so is more complex

###
### new
###
# https://github.com/FluxML/Zygote.jl/blob/master/src/tools/builtins.jl
# https://github.com/FluxML/Zygote.jl/blob/master/src/lib/lib.jl

macro __splatnew__(T, args)
  esc(Expr(:splatnew, T, args))
end

@inline __new__(T, args...) = @__splatnew__(T, args)

grad_mut(x) = Ref{Any}(nt_nothing(x))

struct Jnew{T,G}
  g::G
end

Jnew{T}(g) where T = Jnew{T,typeof(g)}(g)

function pullback(::typeof(__new__), T, args...)
  x = __new__(T, args...)
  g = !ismutabletype(T) || fieldcount(T) == 0 ? nothing : grad_mut(x)
  x, Jnew{T,typeof(g)}(g)
end

@generated function (back::Jnew{T,G})(Δ::Union{NamedTuple,Nothing,RefValue}) where {T,G}
  !ismutabletype(T) && Δ == Nothing && return :nothing
  Δ = G == Nothing ? :Δ :
      Δ <: RefValue ? :(back.g[]) :
      :(accum(back.g[], Δ))
  quote
    x̄ = $Δ
    $(G == Nothing || :(back.g[] = nt_nothing($Δ)))
    (nothing, nothing, $(map(f -> :(x̄.$f), fieldnames(T))...))
  end
end
