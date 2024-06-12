using Test


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
