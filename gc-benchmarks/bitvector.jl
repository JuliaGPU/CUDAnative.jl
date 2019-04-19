module Bitvector

import Base: +, *, <<
using CUDAnative

# This benchmark performs naive arithmetic on bitvectors.
# The goal of the benchmark is to gauge how GPU-unaware
# standard library code that depends on arrays behaves when
# used in a GPU kernel.

const thread_count = 256

@noinline function escape(value)
    Base.pointer_from_objref(value)
    value
end

mutable struct BitInteger{N}
    bits::BitVector
end

function zero(::Type{BitInteger{N}})::BitInteger{N} where N
    BitInteger{N}(falses(N))
end

function one(::Type{BitInteger{N}})::BitInteger{N} where N
    result = falses(N)
    result[1] = true
    return BitInteger{N}(result)
end

function +(a::BitInteger{N}, b::BitInteger{N})::BitInteger{N} where N
    carry = false
    c = falses(N)
    for i in 1:N
        s = Int(a.bits[i]) + Int(b.bits[i]) + Int(carry)
        if s == 1
            carry = false
            c[i] = true
        elseif s == 2
            carry = true
        elseif s == 3
            carry = true
            c[i] = true
        end
    end
    return BitInteger{N}(c)
end

function <<(a::BitInteger{N}, amount::Integer)::BitInteger{N} where N
    c = falses(N)
    for i in 1:(N - amount)
        c[i + amount] = a.bits[i]
    end
    return BitInteger{N}(c)
end

function *(a::BitInteger{N}, b::BitInteger{N})::BitInteger{N} where N
    c = zero(BitInteger{N})
    for i in 1:N
        if a.bits[i]
            c += (b << (i - 1))
        end
    end
    return c
end

function factorial(::Type{BitInteger{N}}, value::Integer)::BitInteger{N} where N
    accumulator = one(BitInteger{N})
    iv = one(BitInteger{N})
    for i in 1:value
        accumulator *= iv
        iv += one(BitInteger{N})
    end
    return accumulator
end

function to_int(value::BitInteger{N})::Int where N
    result = 0
    for i in 1:N
        if value.bits[i]
            result += (1 << (i - 1))
        end
    end
    return result
end

function kernel()
    i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    factorial(BitInteger{128}, 10)
    return
end

end

function bitvector_benchmark()
    # Run the kernel.
    @cuda_sync threads=Bitvector.thread_count Bitvector.kernel()
end

@cuda_benchmark "bitvector" bitvector_benchmark()
