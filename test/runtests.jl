using MicroGrad
using Test

@testset  verbose=true begin
    @testset "reverse" verbose=true begin
        include("reverse.jl")
    end
    @testset "reverse - IR code only" verbose=true begin
        include("reverse_ir.jl")
    end
    @testset "loss functions"  verbose=true begin
        include("loss.jl")
    end
end