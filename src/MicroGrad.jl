module MicroGrad

using IRTools
using Base: RefValue
import Base: show

export rrule, pullback
export sigmoid, tanh_act, softmax, logsoftmax, relu
export mse, logit_cross_entropy, cross_entropy, logsoftmax

include("chainrules.jl")
include("loss.jl")
include("pullback.jl")
include("reverse_ir.jl")
#include("reverse_expr.jl") # alternative to reverse_ir. Less stable, more limited and possible variable pollution.
include("show.jl")

gradient(f, x...) = Base.tail(pullback(f, x...)[2](1))

end