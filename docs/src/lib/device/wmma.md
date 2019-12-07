# WMMA

This section details CUDAnative's interface to CUDA's warp matrix multiply-accumulate (WMMA) operations.
This interface enables programmatic access to Tensor Cores, a new hardware feature in Volta that performs mixed precision matrix MAC operations.

Access to WMMA using CUDAnative is available in two levels: low level wrappers around the LLVM intrinsics, and a higher-level API, similar to that of CUDA C.

Note that to use the WMMA intrinsics, you need a sufficiently recent version of Julia: `v1.4.0-DEV.534` or later.
You can check this by running the following in the REPL:
```julia
VERSION >= v"1.4.0-DEV.534"
```

!!! note

    If you're running into the following error while using the WMMA interfaces:
    ```
    LLVM error: Do not know how to split the result of this operator!
    ```
    then make sure you are running Julia v1.4.0-DEV.534 or later!

## Introduction of Terminology

The WMMA operations perform a matrix multiply-accumulate.
More concretely, it calculates ``D = A \cdot B + C``, where ``A`` is a ``M \times K`` matrix, ``B`` is a ``K \times N`` matrix, and ``C`` and ``D`` are ``M \times N`` matrices.

Note that not all values of ``M``, ``N`` and ``K`` are allowed.
The tuple ``(M, N, K)`` is often called the "shape" of the multiply accumulate operation.

The multiply-accumulate consists of the following steps:
- Load the matrices ``A``, ``B`` and ``C`` from memory to registers using a WMMA load operation.
- Perform the matrix multiply-accumulate of ``A``, ``B`` and ``C`` to obtain ``D`` using a WMMA MMA operation. ``D`` is stored in hardware registers after this step.
- Store the result ``D`` back to memory using a WMMA store operation.

Note that WMMA is a warp-wide operation, which means that all threads in a warp must cooperate, and execute the WMMA operations in lockstep.
Failure to do so will result in undefined behaviour.

Each thread in a warp will hold a part of the matrix in its registers.
In WMMA parlance, this part is referred to as a "fragment".
Note that the exact mapping between matrix elements and fragment is unspecified, and subject to change in future versions.

Finally, it is important to note that the resultant ``D`` matrix can be used as a ``C`` matrix for a subsequent multiply-accumulate.
This is useful if one needs to calculate a sum of the form ``\sum_{i=0}^{n} A_i B_i``, where ``A_i`` and ``B_i`` are matrices of the correct dimension.

## LLVM Intrinsics

The LLVM intrinsics are accessible by using the one-to-one Julia wrappers.
The return type of each wrapper is the Julia type that corresponds closest to the return type of the LLVM intrinsic.
For example, LLVM's `[8 x <2 x half>]` becomes `NTuple{8, NTuple{2, VecElement{Float16}}}` in Julia.
In essence, these wrappers return the SSA values returned by the LLVM intrinsic.
Currently, all intrinsics that are available in LLVM 6, PTX 6.0 and SM 70 are implemented.

These LLVM intrinsics are then lowered to the correct PTX instructions by the LLVM NVPTX backend.
For more information about the PTX instructions, please refer to the [PTX Instruction Set Architecture Manual](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions).

The LLVM intrinsics are subdivided in three categories: load, store and multiply-accumulate.
In what follows, each of these will be discussed.

### Load matrix

**Julia function:** `llvm_wmma_load_{matrix}_{layout}_{shape}_{addr_space}_stride_{elem_type}(src_addr, stride)`

**Corresponding LLVM instrinsic:** `@llvm.nvvm.wmma.load.{matrix}.sync.{layout}.{shape}.{addr_space}.stride.{elem_type}`

**Arguments:**
- `src_addr`: The memory address to load from.
- `stride`: The leading dimension of the matrix, in numbers of elements.

**Placeholders:**
- `{matrix}`: The matrix to load. Can be `a`, `b` or `c`.
- `{layout}`: The storage layout for the matrix. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively.
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point). Note that `f32` is only valid for the matrix ``C``.

### Perform multiply-accumulate

**Julia function:** `llvm_wmma_mma_{a_layout}_{b_layout}_{shape}_{d_elem_type}_{c_elem_type}(a, b, c)`

**Corresponding LLVM instrinsic:** `@llvm.nvvm.wmma.mma.sync.{a_layout}.{b_layout}.{shape}.{d_elem_type}.{c_elem_type}`

**Arguments:**
- `a`: The WMMA fragment corresponding to the matrix ``A``.
- `b`: The WMMA fragment corresponding to the matrix ``B``.
- `c`: The WMMA fragment corresponding to the matrix ``C``.

**Placeholders:**
- `{a_layout}`: The storage layout for matrix ``A``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{b_layout}`: The storage layout for matrix ``B``. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively. Note that this must match the layout used in the load operation.
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{d_elem_type}`: The type of each element in the resultant ``D`` matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).
- `{c_elem_type}`: The type of each element in the ``C`` matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).

!!! warning

    Remember that the shape, type and layout of all operations (be it MMA, load or store) **MUST** match.
    Otherwise, the behaviour is undefined!

### Store matrix

**Julia function:** `llvm_wmma_store_d_{layout}_{shape}_{addr_space}_stride_{elem_type}(dst_addr, data, stride)`

**Corresponding LLVM intrinsic:** `@llvm.nvvm.wmma.store.d.sync.{layout}.{shape}.{addr_space}.stride.{elem_type}`

**Arguments:**
- `dst_addr`: The memory address to store to.
- `data`: The ``D`` fragment to store.
- `stride`: The leading dimension of the matrix, in numbers of elements.

**Placeholders:**
- `{layout}`: The storage layout for the matrix. Can be `row` or `col`, for row major (C style) or column major (Julia style), respectively.
- `{shape}`: The overall shape of the MAC operation. The only valid value is `m16n16k16`.
- `{addr_space}`: The address space of `src_addr`. Can be empty (generic addressing), `shared` or `global`.
- `{elem_type}`: The type of each element in the matrix. Can be `f16` (half precision floating point) or `f32` (full precision floating point).

### Example

````@eval
lines = readlines("../../../../examples/wmma/low-level.jl")
start = findfirst(x -> x == "### START", lines) + 1
stop = findfirst(x -> x == "### END", lines) - 1
example = join(lines[start:stop], '\n')

using Markdown
Markdown.parse("""
```julia
$(example)
```
""")
````

## CUDA C-like API

The main difference between the CUDA C-like API and the lower level wrappers, is that the former enforces several constraints when working with WMMA.
For example, it ensures that the ``A`` fragment argument to the MMA instruction was obtained by a `load_a` call, and not by a `load_b` or `load_c`.
Additionally, it makes sure that the data type and storage layout of the load/store operations and the MMA operation match.

The CUDA C-like API heavily uses Julia's dispatch mechanism.
As such, the method names are much shorter than the LLVM intrinsic wrappers, as most information is baked into the type of the arguments rather than the method name.


Note that, in CUDA C++, the fragment is responsible for both the storage of intermediate results and the WMMA configuration.
All CUDA C++ WMMA calls are function templates that take the resultant fragment as a by-reference argument.
As a result, the type of this argument can be used during overload resolution to select the correct WMMA instruction to call.

In contrast, the API in Julia separates the WMMA storage ([`WMMAFragment`](@ref)) and configuration ([`WMMAConfig`](@ref)).
Instead of taking the resultant fragment by reference, the Julia functions just return it.
This makes the dataflow clearer, but it also means that the type of that fragment cannot be used for selection of the correct WMMA instruction.
Thus, there is still a limited amount of information that cannot be inferred from the argument types, but must nonetheless match for all WMMA operations, such as the overall shape of the MMA.
This is accomplished by a separate "WMMA configuration" (see [`WMMAConfig`](@ref)) that you create once, and then give as an argument to all intrinsics.

### Fragment
```@docs
CUDAnative.WMMAFragmentLayout
CUDAnative.WMMARowMajor
CUDAnative.WMMAColMajor
CUDAnative.WMMAUnspecified
CUDAnative.WMMAFragment
```

### WMMA configuration
```@docs
CUDAnative.WMMAConfig
```

### Load matrix
```@docs
CUDAnative.wmma_load_a
CUDAnative.wmma_load_b
CUDAnative.wmma_load_c
```

### Perform multiply-accumulate
```@docs
CUDAnative.wmma_mma
```

### Store matrix
```@docs
CUDAnative.wmma_store_d
```

### Fill fragment
```@docs
CUDAnative.wmma_fill_c
```

### Example

````@eval
lines = readlines("../../../../examples/wmma/high-level.jl")
start = findfirst(x -> x == "### START", lines) + 1
stop = findfirst(x -> x == "### END", lines) - 1
example = join(lines[start:stop], '\n')

using Markdown
Markdown.parse("""
```julia
$(example)
```
""")
````
