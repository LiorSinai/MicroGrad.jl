# https://github.com/FluxML/IRTools.jl/blob/master/src/reflection/reflection.jl
function meta(T; world=Base.get_world_counter())
    if isnothing(world)
        world = Base.get_world_counter() # in generated function post v1.10 this will return typemax(UInt)
    end
    min_world = Ref{UInt}(typemin(UInt))
    max_world = Ref{UInt}(typemax(UInt))
    has_ambig = Ptr{Int32}(C_NULL)  # don't care about ambiguous results
    _methods = Base._methods_by_ftype(T, #=mt=# nothing, #=lim=# -1,
        world, #=ambig=# false,
        min_world, max_world, has_ambig)
    _methods === nothing && return nothing
    _methods isa Bool && return nothing
    length(_methods) == 0 && return nothing
    last(_methods)
end

xcall(mod::Module, f::Symbol, args...) = Expr(:call, GlobalRef(mod, f), args...)
xcall(f::Symbol, args...) = xcall(Base, f, args...)
xcall(f, args...) = Expr(:call, f, args...)

function returnvalue(ci::Core.CodeInfo)
    for expr in ci.code
        if expr isa Core.ReturnNode
            return expr.val
        end
    end
end

function replace_SSA!(ex::Expr)
    for (i, v) in enumerate(ex.args)
        if v isa Expr
            replace_SSA!(v)
        elseif v isa Core.SSAValue
            ex.args[i] = Symbol("y$(v.id)") 
        end
    end
    ex
end

function replace_slot!(ex::Expr, idx::Int, f::Symbol)
    for (i, v) in enumerate(ex.args)
        if v isa Expr
            replace_slot!(v, idx, f)
        elseif v isa Core.SlotNumber && v.id == idx
            ex.args[i] = :($f) 
        end
    end
    ex
end

function varargs!(ex::Expr, offset::Int=1)
    # assumes all slotnumbers after offset are Vararg.
    # Note that new slotnumbers can be created in the CodeInfo e.g. with an =Expr
    for (i, v) in enumerate(ex.args)
        if v isa Expr
            varargs!(v)
        elseif v isa Core.SlotNumber
            ex.args[i] = :(Base.getindex(args, $(v.id - offset))) 
        end
    end
    ex
end

###
### Pullback
###

# https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/reverse.jl
struct Pullback{S,T}
    data::T
end
  
Pullback{S}(data) where S = Pullback{S,typeof(data)}(data)

function primal(ci::Core.CodeInfo, T=Any)
    tape = []
    calls = []
    ret = []
    for (i, ex) in enumerate(ci.code)
        vy = Symbol("y$i")
        if ex isa Core.ReturnNode
            break
        elseif (typeof(ex) in [Core.GotoNode, Core.GotoIfNot, Core.SlotNumber])
            error("$(typeof(ex)) is not supported")
        elseif (ex isa Expr) && (ex.head == :call)  && !ignored(ex)
            vb = Symbol("back$i")
            new_ex = :(($vy, $vb) = MicroGrad.pullback($(ex.args...)))
            push!(tape, new_ex)
            push!(calls, (;var=vy, expr=ex))
            push!(ret, vb)
        else # keep as is
            push!(tape, :($vy = $ex))
        end
    end
    pb = Expr(:call, Pullback{T}, xcall(:tuple, ret...))
    push!(tape, xcall(:tuple, returnvalue(ci), pb))
    pr = Expr(:block, tape...)
    pr, calls
end

function ignored(ex::Expr)
    f = ex.args[1]
    ignored_f(f)
end

ignored_f(f) = f in (
    GlobalRef(Base, :not_int),
    GlobalRef(Core.Intrinsics, :not_int),
    GlobalRef(Core, :(===)),
    GlobalRef(Core, :apply_type),
    GlobalRef(Core, :typeof),
    GlobalRef(Core, :throw),
    GlobalRef(Base, :kwerr),
    GlobalRef(Core, :kwfunc),
    GlobalRef(Core, :isdefined)
)

function reverse_differentiate(forw::Core.CodeInfo, self, Δ) # Zygote.adjoint
    pr, calls = primal(forw)
    grads = Dict()
    grad!(x, x̄) = push!(get!(grads, x, []), x̄)
    grad(x) = _sum(get(grads, x, [])...)
    grad!(_var_name(returnvalue(forw)), Δ) # _var_name maps to variable names in calls
    tape = Expr[]
    push!(tape, :(data=$(xcall(:getfield, self, QuoteNode(:data)))))
    i = length(calls)
    for (v, ex) in reverse(calls)
        vb = Symbol("back$i")
        push!(tape, :($vb = Base.getindex(data, $i)))
        g = grad(v)
        push!(tape, :(Δs = $vb($g)))
        for (j, x) in enumerate(ex.args)
            xbar = Symbol("x̄$(i)_$(j)")
            get_xbar = :($xbar=$(xcall(:getindex, :Δs, j)))
            push!(tape, get_xbar)
            grad!(_var_name(x), xbar)
        end
        i -= 1
    end
    push!(tape, xcall(:tuple, [grad(x) for x in arguments(forw)]...))
    Expr(:block, tape...)
end

_sum() = nothing
_sum(x) = x
#_sum(x...) = xcall(:+, x...) # only works in simple cases
_sum(xs...) = xcall(MicroGrad, :accum, xs...)

arguments(forw::Core.CodeInfo) = [Symbol("#self"), [Symbol("args$i") for i in 2:length(forw.slotnames)]...]
_var_name(x::Core.SlotNumber) = x.id == 1 ? Symbol("#self") : Symbol("args$(x.id)")
_var_name(x::Core.SSAValue)  = Symbol("y$(x.id)")
_var_name(x) = x

###
### @generated forward
###

function _generate_pullback(world, f, args...)
    T = Tuple{f, args...}
    if (has_chain_rrule(T, world))
        return :(rrule(f, args...))
    end    
    pr, backs = _generate_pullback_by_decomposition(T, world)
    replace_slot!(pr, 1, :f)
    varargs!(pr)
    replace_SSA!(pr)
    pr
end

# https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/chainrules.jl
function has_chain_rrule(T, world)
    Tr = Tuple{typeof(rrule), T.parameters...}
    meta_T = meta(Tr; world=world)
    if isnothing(meta_T)
        return false
    end
    type_signature, sps, method_ = meta_T
    method_.sig.parameters[2] !== Any
end

function _generate_pullback_by_decomposition(T, world)
    m = meta(T; world=world)
    isnothing(m) && return :(error("No method found for ", repr($T), " in world ", $world))
    type_signature, sps, method_ = m
    ci = Base.uncompressed_ast(method_)
    pr, calls = primal(ci, T)
end

###
### @generated reverse
### 
function _generate_callable_pullback(j::Type{<:Pullback{T, S}}, world, Δ) where {T, S}
    m = meta(T; world=world)
    isnothing(m) && return :(error("No method found for ", repr($T), " in world ", $world))
    type_signature, sps, method_ = m
    ci = Base.uncompressed_ast(method_)
    back = reverse_differentiate(ci, :methodinstance, :Δ)
    back
end 

### @generated functions

if VERSION >= v"1.10.0-DEV.873"
    function _pullback_generator(world::UInt, source, self, f, args)
        ret = _generate_pullback(world, f, args...)
        ret isa Core.CodeInfo && return ret
        stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :f, :args), Core.svec())
        stub(world, source, ret)
    end
      
    @eval function pullback(f, args...)
        $(Expr(:meta, :generated, _pullback_generator))
        $(Expr(:meta, :generated_only))
    end

    function _callable_pullback_generator(world::UInt, source, self, Δ)
        ret = _generate_callable_pullback(self, world, Δ)
        ret isa Core.CodeInfo && return ret
        stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :Δ), Core.svec()) # names must match symbols in _generate_callable_pullback
        stub(world, source, ret)
    end

    @eval function (j::Pullback)(Δ)
        $(Expr(:meta, :generated, _callable_pullback_generator))
        $(Expr(:meta, :generated_only))
    end
else
    @generated function pullback(f, args...)
        _generate_pullback(nothing, f, args...)
    end

    @generated function (methodinstance::Pullback)(Δ) # argument names must match symbols in _generate_callable_pullback
        _generate_callable_pullback(methodinstance, nothing, Δ)
    end
end
