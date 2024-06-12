@testset "activations" begin
    z, back = pullback(sigmoid, 0.5)
    @test isapprox(z, 0.6224593312018546)
    grad_ = back(1.0)
    @test isapprox(grad_[2], 0.2350037122015945)

    z, back = pullback(tanh_act, 0.5)
    @test isapprox(z, 0.46211715726000974)
    grad_ = back(1.0)
    @test isapprox(grad_[2], 0.7864477329659274)
end

@testset "activations broadcast" begin
    A = [
        -1.2  0.5
         2.1 -0.5
    ]

    Z, back = pullback(sigmoid, A) # Tuple{typeof(sigmoid), Matrix{Float64}}
    grad_ = back(1.0)
    grad_expected = [
        0.177894   0.235004
        0.0971947  0.235004
    ]
    @test isapprox(grad_[2], grad_expected, atol=1e-6)

    Z, back = pullback(x -> tanh_act(x), A)
    grad_expected = [
        0.30502   0.786448
        0.058223  0.786448
    ]
    grad_ = back(1.0)
    @test isapprox(grad_[2], grad_expected, atol=1e-6)
end

@testset "softmax" begin
    x = [2.0, -1.0, 0.5]

    z, back = pullback(softmax, x)
    expected = [ 0.7855970345892759, 0.03911257327068745, 0.1752903921400367 ]
    @test isapprox(z, expected)
    grad_ = back(ones(3))
    @test grad_[2] == [0.0, 0.0, 0.0]
    grad_ = back([0.46, .989, 0.68])
    expected_grad = (nothing, [-0.04655011041061041, 0.01837295773064898, 0.028177152679961473])
    @test isapprox(grad_[2], expected_grad[2])

    z, back = pullback(logsoftmax, x)
    expected = [ -0.24131129665715703, -3.241311296657157, -1.7413112966571571]
    @test isapprox(z, expected)
    grad_ = back(ones(3))
    expected_grad = (nothing, [-1.3567911037678275, 0.8826622801879377, 0.47412882357988995])
    @test isapprox(grad_[2], expected_grad[2])

    X = [
        2.0  2.0
       -1.0  0.4
        0.5  9.0
    ]

    Z, back = pullback(softmax, X)
    expected = [ 
        0.785597   0.000910884
        0.0391126  0.000183904
        0.17529    0.998905
    ]
    @test isapprox(Z, expected; atol=1e-6)
    grad_ = back(ones(3, 2))
    @test grad_[2] == [0.0 0.0; 0.0 0.0; 0.0 0.0]
    grad_ = back([0.46 -0.95; 0.989 -8.6; 0.68 6.3])
    expected_grad = (nothing, [
        -0.0465501  -0.0065954
        0.018373   -0.00273845
        0.0281772   0.00933385
    ])
    @test isapprox(grad_[2], expected_grad[2]; atol=1e-6)

    Z, back = pullback(logsoftmax, X)
    expected = [
        -0.241311  -7.0011
        -3.24131   -8.6011
        -1.74131   -0.00109539
    ]
    @test isapprox(Z, expected; atol=1e-5)
    grad_ = back(ones(3, 2))
    expected_grad = (
        nothing, 
        [   -1.3567911037678275 0.997267349055322;
            0.8826622801879377 0.9994482872893747;
            0.47412882357988995 -1.996715636344697
        ]
    )
    @test isapprox(grad_[2], expected_grad[2])
end

@testset "loss" begin
    ŷ = [4.0, 2.0, -3.0, 1.5]
    y = [6.0, -2.0, -1.0, 1.5]

    z, back = pullback(mse, ŷ, y)
    @test z == 6.0
    grad_ = back(1.0)
    @test grad_ == (nothing, [-1.0, 2.0, -1.0, 0.0], [1.0, -2.0, 1.0, -0.0])

    z, back = pullback(logit_cross_entropy, ŷ, y)
    @test isapprox(z, -6.3613271992582145)
    grad_ = back(1.0)
    expected_grad = (
        nothing,
        [-2.306426003013708, 2.4998708830375276, 1.003368103516293, -1.1968129835401118],
        [0.1974828446092856, 2.1974828446092856, 7.197482844609286, 2.6974828446092856]
    )
    @test isapprox(grad_[2], expected_grad[2])
    @test isapprox(grad_[3], expected_grad[3])

    Y = [
        1.0 0.0 1.0
        0.0 1.0 0.0
    ]
    Ŷ = [
        -0.5  0.6 -1.0
         1.0  0.3  0.4
    ]
    z, back = pullback(logit_cross_entropy, Ŷ, Y)
    @test isapprox(z, 1.3920619774565768)
    grad_ = back(1.0)
    expected_grad = (
        nothing, 
        [-0.2725248253978812 0.19148083893721968 -0.2673946295195272; 0.2725248253978812 -0.19148083893721962 0.26739462951952725], 
        [0.5671377593275841 0.184785081489509 0.5401391366394835; 0.06713775932758415 0.284785081489509 0.07347246997281694]
    )
    @test isapprox(grad_[2], expected_grad[2])
    @test isapprox(grad_[3], expected_grad[3])
end
