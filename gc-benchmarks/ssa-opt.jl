# This benchmark defines a simple SSA IR, creates a basic
# block on the GPU and applies the constant folding optimization
# to it.

module SSAOpt

# A base type for SSA instructions.
abstract type Instruction end

# A base type for values or flow in an SSA basic block.
abstract type ValueOrFlow end

# A value in an SSA control-flow graph.
mutable struct Value <: ValueOrFlow
    # The instruction that computes the value.
    instruction::Instruction

    # The next value or control-flow instruction.
    next::ValueOrFlow
end

# A base type for control-flow instructions in an SSA basic block.
abstract type Flow <: ValueOrFlow end

# A control-flow instruction that returns a value.
mutable struct ReturnFlow <: Flow
    # The value to return.
    result::Value
end

# A control-flow instruction that represents undefined control flow.
mutable struct UndefinedFlow <: Flow end

# A basic block in an SSA control-flow graph.
mutable struct BasicBlock
    # The first value or flow instruction in the basic block.
    head::ValueOrFlow
end

# An integer constant instruction.
mutable struct IConst <: Instruction
    value::Int
end

# An integer addition instruction.
mutable struct IAdd <: Instruction
    # The left value.
    left::Value
    # The right value.
    right::Value
end

# Folds constants in a basic block.
function fold_constants(block::BasicBlock)
    value = block.head
    while isa(value, Value)
        insn = value.instruction
        if isa(insn, IAdd)
            left = insn.left.instruction
            right = insn.right.instruction
            if isa(left, IConst)
                if isa(right, IConst)
                    value.instruction = IConst(left.value + right.value)
                end
            end
        end
        value = value.next
    end
    block
end

# Creates a block that naively computes `sum(1:range_max)`.
function create_range_sum_block(range_max)
    head = accumulator = Value(IConst(0), UndefinedFlow())
    for i in 1:range_max
        constant = Value(IConst(i), UndefinedFlow())
        accumulator.next = constant
        accumulator = Value(IAdd(accumulator, constant), UndefinedFlow())
        constant.next = accumulator
    end
    ret_flow = ReturnFlow(accumulator)
    accumulator.next = ret_flow
    BasicBlock(head)
end

const thread_count = 256

function kernel()
    block = create_range_sum_block(50)
    fold_constants(block)
    return
end

end

function ssaopt_benchmark()
    @cuda_sync threads=SSAOpt.thread_count SSAOpt.kernel()
end

@cuda_benchmark "ssa opt" ssaopt_benchmark()
