# based on https://github.com/FluxML/IRTools.jl/blob/master/examples/reverse.jl

using IRTools
using IRTools: arguments, block, blocks, finish, IR, isexpr, meta, Pipe,
    return!, returnvalue, stmt, xcall
using IRTools.Inner: argument!, inlineable!, pis!, slots!, update!, varargs!, Variable, dummy_m

###
### Helper functions
###

function build_codeinfo_(ir::IR)
    ir = copy(ir)
    ci = Base.uncompressed_ir(dummy_m)
    ci.inlineable = true
    for arg in arguments(ir)
    @static if VERSION >= v"1.10.0-DEV.870"
        isnothing(ci.slottypes) && (ci.slottypes = Any[])
        push!(ci.slottypes, Type)
    end
    push!(ci.slotnames, Symbol(""))
    push!(ci.slotflags, 0)
    end
    #argument!(ir, at = 1) # argument for #self# might already exist
    update!(ci, ir)
end

iscall(x, m::Module, n::Symbol) = isexpr(x, :call) && x.args[1] == GlobalRef(m, n)

# Hack to work around fragile constant prop through overloaded functions
unwrapquote(x) = x
unwrapquote(x::QuoteNode) = x.value

###
### Pullback
###

# https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/reverse.jl

struct Pullback{S,T}
    data::T
end
  
Pullback{S}(data) where S = Pullback{S,typeof(data)}(data)

function primal(ir::IR, T=Any)
    pr = Pipe(ir) # make inserts into ir efficient
    calls = []
    pullbacks = []
    for (v, st) in pr
        ex = st.expr
        if isexpr(ex, :call) && !ignored(ex)
            t = insert!(pr, v, stmt(xcall(MicroGrad, :pullback, ex.args...), line = st.line))
            pr[v] = xcall(Base, :getindex, t, 1)
            J = push!(pr, xcall(:getindex, t, 2))
            push!(calls, v)
            push!(pullbacks, J)
        end
    end
    pb = Expr(:call, Pullback{T}, xcall(:tuple, pullbacks...))
    return!(pr, xcall(:tuple, returnvalue(block(ir, 1)), pb))
    finish(pr), calls
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

function instrument(ir::IR)
    pr = Pipe(ir)
    for (v, st) in pr
        ex = st.expr
        if isexpr(ex, :new)
            pr[v] = xcall(MicroGrad, :__new__, ex.args...)
        elseif is_literal_getfield(ex)
            pr[v] = xcall(MicroGrad, :literal_getfield, ex.args[2], Val(unwrapquote(ex.args[3])))
        end
    end
    ir = finish(pr)
    ir
end

is_literal_getfield(ex) =
  (iscall(ex, Core, :getfield) || iscall(ex, Base, :getfield)) &&
  ex.args[3] isa Union{QuoteNode,Integer}

function reverse_differentiate(forw::IR) # Zygote.adjoint
    ir = empty(forw)
    grads = Dict()
    grad!(x, x̄) = push!(get!(grads, x, []), x̄)
    grad(x) = _sum(get(grads, x, [])...)
    self = argument!(ir, at = 1, insert=false)
    grad!(returnvalue(block(forw, 1)), IRTools.argument!(ir)) # starting gradient is incoming Δ
    data = push!(ir, xcall(:getfield, self, QuoteNode(:data)))
    _, calls = primal(forw)
    pullbacks = Dict(calls[i] => push!(ir, xcall(:getindex, data, i)) for i = 1:length(calls))
    for v in reverse(keys(forw))
        ex = forw[v].expr
        if isexpr(ex, :call) && !ignored(ex)
            Δs = push!(ir, Expr(:call, pullbacks[v], grad(v)))
            for (i, x) in enumerate(ex.args)
                grad!(x, push!(ir, xcall(:getindex, Δs, i)))
            end
        end
    end
    return!(ir, xcall(:tuple, [grad(x) for x in arguments(forw)]...))
end

_sum() = nothing
_sum(x) = x
_sum(xs...) = xcall(MicroGrad, :accum, xs...)

###
### @generated forward
###
function _generate_pullback(world, f, args...)
    T = Tuple{f, args...}
    if (has_chain_rrule(T, world))
        return :(rrule(f, args...))
    end    
    g = _generate_pullback_by_decomposition(T, world)
    if isnothing(g)
        return :(error("No method found for ", repr($T), " in world ", $world))
    end
    m, pr, backs =  g
    pr = varargs!(m, pr, 1)
    pr = slots!(pis!(pr))
    argument!(pr, at = 1) # add #self#
    ci = build_codeinfo_(pr)
    ci.slotnames = [Symbol("#self#"), :f, :args]
    ci
end

# https://github.com/FluxML/Zygote.jl/blob/master/src/compiler/chainrules.jl
function has_chain_rrule(T, world)
    Tr = Tuple{typeof(rrule), T.parameters...}
    meta_T = meta(Tr; world=world)
    if isnothing(meta_T)
        return false
    end
    method_ = meta_T.method
    sig = method_.sig
    !(sig isa DataType) || (sig.parameters[2] !== Any)
end

function _generate_pullback_by_decomposition(T, world)
    m = meta(T; world=world)
    isnothing(m) && return nothing
    ir = IR(m)
    length(blocks(ir)) == 1 || error("control flow is not supported")
    ir = instrument(ir)
    pr, calls = primal(ir, T)
    m, pr, calls
end

###
### @generated reverse
### 
function _generate_callable_pullback(j::Type{<:Pullback{S, T}}, world, Δ) where {S, T}
    m = meta(S; world=world)
    ir = IR(m)
    isnothing(ir) && return :(error("Non-differentiable function ", repr(args[1])))
    length(blocks(ir)) == 1 || error("control flow is not supported")
    ir = instrument(ir)
    back = reverse_differentiate(ir)
    back = slots!(inlineable!(back))
    ci = build_codeinfo_(back)
    ci.slotnames = [Symbol("#self#"), :Δ]
    ci
end 

### @generated functions

if VERSION >= v"1.10.0-DEV.873"
    # on Julia 1.10, generated functions need to keep track of the world age
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
        stub = Core.GeneratedFunctionStub(identity, Core.svec(:methodinstance, :Δ), Core.svec())
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

    @generated function (j::Pullback)(Δ)
        _generate_callable_pullback(j, nothing, Δ)
    end
end
