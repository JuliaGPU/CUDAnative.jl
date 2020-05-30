export Transform
module Transform

# ---------------------
# Elementwise transform
# ---------------------

export Elementwise

"""
    Elementwise{F}

A simple transformation that applies a function elementwise.

# Example
```julia
double_elements = Elementwise(x -> x * 2)
```
"""
struct Elementwise{F}
    func::F
end

@inline Elementwise() = Elementwise(identity)

@inline (transf::Elementwise)(x, tile) = transf.func.(x)

end
