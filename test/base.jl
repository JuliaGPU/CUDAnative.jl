@testset "base interface" begin

############################################################################################

@testset "method caching" begin

import InteractiveUtils: _dump_function

# #17057 fallout
@noinline post17057_child(i) = sink(i)
function post17057_parent(arr::Ptr{Int64})
    i = post17057_child(0)
    unsafe_store!(arr, i, i)
end

# bug: default module activation segfaulted on NULL child function if cached=false

if VERSION >= v"1.5.0-DEV.423"
    params = Base.CodegenParams()
else
    params = Base.CodegenParams(cached=false)
end

if VERSION >= v"1.1.0-DEV.762"
    _dump_function(post17057_parent, Tuple{Ptr{Int64}},
                   #=native=#false, #=wrapper=#false, #=strip=#false,
                   #=dump_module=#true, #=syntax=#:att, #=optimize=#false, :none,
                   params)
else
    _dump_function(post17057_parent, Tuple{Ptr{Int64}},
                   #=native=#false, #=wrapper=#false, #=strip=#false,
                   #=dump_module=#true, #=syntax=#:att, #=optimize=#false,
                   params)
end

end

############################################################################################

end
