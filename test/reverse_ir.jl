using MicroGrad
using Test

@assert MicroGrad.AD_MODE == "IR"

@testset "not supported" begin
    foo(x::Number) = x > 0 ? x : 0 # relu
    @test_throws "control flow is not supported" pullback(foo, -5)

    function pow(x::Number, n::Int)
        r = one(x)
        while n > 0
            n -= 1
            r *= x
        end
        r
    end
    @test_throws "control flow is not supported" pullback(pow, 3.2, 5)
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
    z, back = pullback(confuse, 3.0)
    @test_throws MethodError back(1.0)
#end

@testset "generated" begin    
    @generated function _apply_chain(layers::Tuple{Vararg{Any,N}}, x) where {N}
      symbols = vcat(:x, [gensym() for _ in 1:N])
      calls = [:($(symbols[i+1]) = layers[$i]($(symbols[i]))) for i in 1:N]
      Expr(:block, calls...)
    end

    grad_chain(x) = -cos(cos(x)) * sin(x)
    chain = (cos, sin)
    z, back = pullback(_apply_chain, chain, 0.6)
    @test z ≈ 0.7347754747082689
    grad_ = back(1.0)
    @test grad_[3] ≈ grad_chain(0.6)
end

@testset "model map" begin
    struct TestModelMap{V<:AbstractVector}
        weights::V
    end
    
    (m::TestModelMap)(x) = evalpoly(x, m.weights)
    (m::TestModelMap)(x::AbstractVector) = map(x -> evalpoly(x, m.weights), x)

    model = TestModelMap([3.0, 2.0, -3.0, 1.0])

    x, y = [1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 3.0, 15.0]
    z, back = pullback(model, x) # T = Tuple{typeof(model), typeof(x)}
    @test z == [3.0, 3.0, 9.0, 27.0]
    expected = ((weights = [4.0, 10.0, 30.0, 100.0],), [-1.0, 2.0, 11.0, 26.0])
    grad_ = back(ones(4))
    @test grad_ == expected

    function grad_loss_xy(x::Number, y::Number, n::Int)
        c = 2/n * (model(x) - y)
        Δy = -c
        Δx = grad_model(x)[2] * c
        (nothing, Δx, Δy)
    end
    function grad_loss_xy(xs::Vector, ys::Vector)
        grad_s = map((x, y) -> grad__loss(x, y, length(ys)), xs, ys)
        Δx = map(g->g[2], grad_s)
        Δy = map(g->g[3], grad_s)
        (nothing, Δx, Δy)
    end

    z, back = pullback((x, y) -> mse(model(x), y), x, y) # note: this will also make a closure around model
    @test z == 46.25
    grad_ = back(1.0)
    @test grad_[2] == [-0.5, 2.0, 33.0, 156.0]
    @test grad_[3] == [-0.5, -1.0, -3.0, -6.0]

    z, back = pullback(m-> mse(m(x), y), model) # note: this will also make a closure around (x, y)
    @test z == 46.25
    grad_ = back(1.0)
    @test grad_[2] == (weights = [10.5, 35.5, 127.5, 473.5],)
end
