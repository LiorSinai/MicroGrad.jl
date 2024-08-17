using MicroGrad
using Test

@assert MicroGrad.AD_MODE == "EXPRESSION"

@testset "not supported" begin
    foo(x::Number) = x > 0 ? x : 0 # relu
    @test_throws "Core.GotoIfNot is not supported" pullback(foo, -5)

    function pow(x::Number, n::Int)
        r = one(x)
        while n > 0
            n -= 1
            r *= x
        end
        r
    end
    @test_throws "Core.GotoIfNot is not supported" pullback(pow, 3.2, 5)
end

#@testset "confused" begin
# fails when under @testset
function confuse(x)
    y = x * 3
    z = y + 2
    b = z * x
    y
end
x = confuse(3.0)
@test_throws ErrorException pullback(confuse, 3.0)
#end

@testset "generated" begin    
    @generated function _apply_chain(layers::Tuple{Vararg{Any,N}}, x) where {N}
      symbols = vcat(:x, [gensym() for _ in 1:N])
      calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
      Expr(:block, calls...)
    end

    chain = (cos, sin)
    @test_throws "Method is @generated; try `code_lowered` instead." pullback(_apply_chain, chain, 0.6)
end

@testset "model map" begin
    struct TestModelMap{V<:AbstractVector}
        weights::V
    end
    
    (m::TestModelMap)(x) = evalpoly(x, m.weights)
    (m::TestModelMap)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)

    model = TestModelMap([3.0, 2.0, -3.0, 1.0])

    x, y = [1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 15.0]
    @test_throws "Core.SlotNumber is not supported" pullback(model, x)
end
