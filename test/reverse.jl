using Test

@testset "basic" begin
    z, back = pullback(sin, 0.6)
    @test z ≈ 0.5646424733950354
    grad_ = back(1.0)
    @test isnothing(grad_[1])
    @test grad_[2] == cos(0.6)
end

@testset "trig chainrule" begin
    foo(x) = sin(cos(x))
    grad_foo(x) = - cos(cos(x)) * sin(x)

    z, back = pullback(foo, 0.6)
    @test z ≈ 0.7347754747082689
    grad_ = back(1.0)
    @test isnothing(grad_[1])
    @test grad_[2] ≈ grad_foo(0.6)
end

@testset "arrays" begin
    A = [
        4.4 6.9 6.2 8.4
        9.2 7.4 8.0 0.3
        0.6 0.4 0.5 0.7
    ]
    X = [
        1.1 6.5
        3.7 8.5
        9.8 6.6
        6.1 2.4
    ]

    Y, back = pullback(*, A, X)
    expected = [
        142.37 148.33
        117.73 176.22
         11.31  12.28
    ]
    @test isapprox(Y, expected; atol=1e-6)
    Δ = [
        5.6 5.3
        2.8 6.9
        8.9 1.0
    ]
    grad_ = back(Δ)
    @test isnothing(grad_[1])
    @test grad_[2] == Δ * X'
    @test grad_[3] ==  A' * Δ
end

@testset "array broadcast" begin
    # broadcast
    A = [
        4.4 6.9 6.2 8.4
        9.2 7.4 8.0 0.3
        0.6 0.4 0.5 0.7
    ]
    X = [
        1.1 6.5
        3.7 8.5
        9.8 6.6
        6.1 2.4
    ]
    b = [10; 20; 30]
    Y, back = pullback((x, y) -> x .+ y, A, b)
    expected =[
        14.4 16.9 16.2 18.4
        29.2 27.4 28.0 20.3
        30.6 30.4 30.5 30.7
    ]
    @test isapprox(Y, expected; atol=1e-6)
    Δ = [
        10.0 11.0 12.0 13.0
        20.0 21.0 22.0 23.0
        30.0 31.0 32.0 33.0
    ]
    grad_ = back(Δ)
    @test isnothing(grad_[1])
    @test grad_[2] == Δ
    @test grad_[3] ==  [46; 86.0; 126.0;;]

    # combined
    dense(X::AbstractVecOrMat, W::AbstractMatrix, b::AbstractVector) = W * X .+ b
    Y, back = pullback(dense, X, A, b)
    expected = [
        152.37 158.33
        137.73 196.22
         41.31 42.28
    ]
    @test isapprox(Y, expected; atol=1e-6)
    Δ = [
        10.0 11.0
        20.0 21.0
        30.0 31.0
    ]
    grad_ = back(Δ)
    @test isnothing(grad_[1])
    @test grad_[2] == A' * Δ # ΔX
    @test grad_[3] == Δ * X' # ΔA
    @test grad_[4] == [21 ; 41; 61 ;;]
end

@testset "reuse variable" begin
    bar(a, b) = a / (a + b*b) # T = Tuple{typeof(bar), Float64, Float64}
    grad_bar(a, b) = (nothing, b*b/(a + b*b)^2, -2*a*b/(a + b*b)^2)

    z, back = pullback(bar, 2.1, 3.2)
    @test z ≈ 0.17017828200972446
    grad_ = back(1.0)
    @test isnothing(grad_[1])
    expected = grad_bar(2.1, 3.2)
    @test grad_[2] ≈ expected[2]
    @test grad_[3] ≈ expected[3]
end

@testset "map" begin
    x = [0.1, 0.2, 0.5]
    z, back = pullback(map, sin, x) 
    grad_ = back([1.0, 1.0, 1.0])
    @test isnothing(grad_[1])
    @test isnothing(grad_[2])
    @test grad_[3] ≈ [0.9950041652780258, 0.9800665778412416, 0.8775825618903728]

    bar(a, b) = a / (a + b*b) # T = Tuple{typeof(bar), Float64, Float64}
    grad_bar(a, b) = (nothing, b*b/(a + b*b)^2, -2*a*b/(a + b*b)^2)

    z, back = pullback(map, bar, [2.1, 4.0], [3.2, 5.0])
    grad_ = back([1.0, 1.0])
    @test isnothing(grad_[1])
    @test isnothing(grad_[2])
    @test grad_[3] ≈ [0.06724649254378245, 0.029726516052318668]
    @test grad_[4] ≈ [-0.08826102146371445, -0.047562425683709865]
end

@testset "getfield" begin
    struct Boo
        x
        y
        z
    end
    boo = Boo(1, "aaa", [1, 2, 3])

    z, back = pullback(getfield, boo, :x)
    @test z == 1
    expected = (nothing, (x = 1.0, y = nothing, z = nothing), nothing)
    grad_ = back(1.0)
    @test grad_ == expected

    z, back = pullback(getfield, boo, :z)
    @test z == [1, 2, 3]
    expected = (nothing, (x = nothing, y = nothing, z = 1.0), nothing)
    grad_ = back(1.0)
    @test grad_ == expected
end

@testset "closure" begin
    function closure_pullback(a::Number, b::Number, x::Number)
        f(x) = a * x + b
        z, back = pullback(f, x)
        z, back, f
    end
    z, back, g = closure_pullback(2.0, 3.5, 4.0)
    @test z == 11.5
    grad_ = back(1.0)
    expected = ((a = 4.0, b = 1.0), 2.0)
    @test grad_ == expected
end

@testset "model" begin
    struct TestModel{V<:AbstractVector}
        weights::V
    end
    
    (m::TestModel)(x) = evalpoly(x, m.weights)

    model = TestModel([3.0, 2.0, -3.0, 1.0])
    
    grad_model(x::Number) = ((weights = [1.0, x, x^2, x^3],), 2.0 - 6*x + 3x^2) 
    function grad_model(x::Vector)
        grad_s = map(grad_model, x)
        Δweights = sum(map(g->g[1].weights, grad_s))
        Δx = map(g->g[2], grad_s)
        ((weights=Δweights,), Δx)
    end

    z, back = pullback(model, 2.5)
    @test z == 4.875
    expected = ((weights = [1.0, 2.5, 6.25, 15.625],), 5.75)
    grad_ = back(1.0)
    @test grad_ == expected
end
