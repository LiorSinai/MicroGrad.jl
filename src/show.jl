funcname(::Type{Type{T}}) where T = string(T)

function funcname(T)
  if isdefined(T, :instance)
    string(T.instance)
  else
    "λ"
  end
end

function Base.show(io::IO, j::Pullback{S}) where S
  name = funcname(S.parameters[1])
  print(io, "∂($name)")
  #print(io, j.data)
end
