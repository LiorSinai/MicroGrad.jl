# based on https://scikit-learn.org/stable/modules/classes.html#samples-generator
using Random

function make_moons(rng::AbstractRNG, n_samples::Int=100; noise::Union{Nothing, AbstractFloat}=nothing)
    n_moons = floor(Int, n_samples / 2)
    t_min = 0.0
    t_max = π
    t_inner = rand(rng, n_moons) * (t_max - t_min) .+ t_min
    t_outer = rand(rng, n_moons) * (t_max - t_min) .+ t_min
    outer_circ_x = cos.(t_outer)
    outer_circ_y = sin.(t_outer)
    inner_circ_x = 1 .- cos.(t_inner)
    inner_circ_y = 1 .- sin.(t_inner) .- 0.5

    data = [outer_circ_x outer_circ_y; inner_circ_x inner_circ_y]
    z = permutedims(data, (2, 1))
    if !isnothing(noise)
        z += noise * randn(size(z))
    end
    z
end

make_moons(n_samples::Int=100; options...) = make_moons(Random.default_rng(), n_samples; options...)
