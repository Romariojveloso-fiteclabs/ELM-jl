mutable struct ELMFitAccumulator
    gram::Matrix{Float64}
    rhs::Matrix{Float64}
    class_to_index::Dict{Any,Int}
    rows::Int
    class_counts::Vector{Int}
end

function initialize_batched_fit!(
    model::ELMClassifier{T},
    input_size::Integer,
    classes::AbstractVector,
) where {T}
    input_size > 0 || throw(ArgumentError("input_size must be positive"))
    class_values = collect(classes)
    length(class_values) >= 2 || throw(ArgumentError("at least two classes are required"))
    length(unique(class_values)) == length(class_values) || throw(ArgumentError(
        "classes must be unique",
    ))

    rng = MersenneTwister(model.seed)
    model.input_size = Int(input_size)
    model.input_weights = randn(rng, T, model.input_size, model.hidden_neurons)
    model.biases = randn(rng, T, model.hidden_neurons)
    model.output_weights = Matrix{T}(undef, 0, 0)
    model.classes = class_values
    return ELMFitAccumulator(
        zeros(Float64, model.hidden_neurons, model.hidden_neurons),
        zeros(Float64, model.hidden_neurons, length(class_values)),
        Dict{Any,Int}(class => index for (index, class) in enumerate(class_values)),
        0,
        zeros(Int, length(class_values)),
    )
end

function partial_fit!(
    accumulator::ELMFitAccumulator,
    model::ELMClassifier{T},
    X::AbstractMatrix,
    y::AbstractVector,
) where {T}
    size(X, 1) == length(y) || throw(DimensionMismatch("X and y batch sizes differ"))
    size(X, 2) == model.input_size || throw(DimensionMismatch(
        "X has $(size(X, 2)) features; expected $(model.input_size)",
    ))
    isempty(y) && return accumulator

    features = _checked_features(T, X)
    hidden = model.activation.(features * model.input_weights .+ model.biases')
    targets = zeros(T, length(y), length(model.classes))
    for (row, label) in enumerate(y)
        index = get(accumulator.class_to_index, label, 0)
        iszero(index) && throw(ArgumentError("unknown class in batch: $(repr(label))"))
        targets[row, index] = one(T)
        accumulator.class_counts[index] += 1
    end
    accumulator.gram .+= Float64.(hidden' * hidden)
    accumulator.rhs .+= Float64.(hidden' * targets)
    accumulator.rows += length(y)
    return accumulator
end

function finalize_batched_fit!(
    accumulator::ELMFitAccumulator,
    model::ELMClassifier{T},
) where {T}
    accumulator.rows > 0 || throw(ArgumentError("no batches were accumulated"))
    all(>(0), accumulator.class_counts) || throw(ArgumentError(
        "every configured class must occur in the accumulated batches",
    ))
    system = copy(accumulator.gram)
    for index in axes(system, 1)
        system[index, index] += Float64(model.lambda)
    end
    model.output_weights = Matrix{T}(system \ accumulator.rhs)
    return model
end
