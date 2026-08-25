module ELM

using LinearAlgebra
using Random
import Base: precision

include("activations.jl")
include("preprocessing.jl")
include("iot23.jl")
include("metrics.jl")

export ELMClassifier, fit!, predict, predict_scores, binary_decision_scores,
       predict_with_threshold, optimal_f1_threshold,
       ELMFitAccumulator, initialize_batched_fit!, partial_fit!, finalize_batched_fit!,
       abs_activation, tanh_activation,
       load_dataset, normalize_features, NormalizationParams,
       encode_labels, train_test_split,
       IoT23Preprocessor, fit_iot23_preprocessor, fit_all_iot23_preprocessors,
       foreach_iot23_batch, save_iot23_preprocessor,
       load_iot23_preprocessor, IoT23SamplingPlan,
       balanced_iot23_sampling_plan, foreach_sampled_iot23_batch,
       accuracy, precision, recall, f1_score

"""A single-hidden-layer Extreme Learning Machine classifier."""
mutable struct ELMClassifier{T<:AbstractFloat,F}
    hidden_neurons::Int
    activation::F
    lambda::T
    seed::Int
    input_weights::Matrix{T}
    biases::Vector{T}
    output_weights::Matrix{T}
    classes::Any
    input_size::Int
end

"""Sufficient statistics for bounded-memory ELM training."""
mutable struct ELMFitAccumulator
    gram::Matrix{Float64}
    rhs::Matrix{Float64}
    class_to_index::Dict{Any,Int}
    rows::Int
    class_counts::Vector{Int}
end

function ELMClassifier(;
    hidden_neurons::Integer=100,
    activation::F=abs_activation,
    datatype::Type{T}=Float64,
    lambda::Real=0.01,
    seed::Integer=42,
) where {T<:AbstractFloat,F}
    hidden_neurons > 0 || throw(ArgumentError("hidden_neurons must be positive"))
    lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
    return ELMClassifier{T,F}(
        Int(hidden_neurons), activation, T(lambda), Int(seed),
        Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
        Matrix{T}(undef, 0, 0), nothing, 0,
    )
end

"""
    initialize_batched_fit!(model, input_size, classes)

Initialize reproducible hidden-layer parameters and return an accumulator for
`H' * H` and `H' * Y`. Call `partial_fit!` for each batch and
`finalize_batched_fit!` once at the end.
"""
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

"""Accumulate one transformed training batch without retaining its hidden matrix."""
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

"""Solve the accumulated regularized system and store the output weights."""
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

function _checked_features(::Type{T}, X::AbstractMatrix) where {T<:AbstractFloat}
    isempty(X) && throw(ArgumentError("X must not be empty"))
    converted = try
        Matrix{T}(X)
    catch error
        throw(ArgumentError("all features in X must be numeric and convertible to $T: $error"))
    end
    all(isfinite, converted) || throw(ArgumentError("X contains NaN or infinite values"))
    return converted
end

"""Fit `model` to a samples-by-features matrix `X` and label vector `y`."""
function fit!(model::ELMClassifier{T}, X::AbstractMatrix, y::AbstractVector) where {T}
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but y has $(length(y)) labels",
    ))
    isempty(y) && throw(ArgumentError("y must not be empty"))

    features = _checked_features(T, X)
    targets, classes = encode_labels(y; datatype=T)
    rng = MersenneTwister(model.seed)

    model.input_size = size(features, 2)
    model.input_weights = randn(rng, T, model.input_size, model.hidden_neurons)
    model.biases = randn(rng, T, model.hidden_neurons)
    hidden_output = model.activation.(features * model.input_weights .+ model.biases')
    gram = hidden_output' * hidden_output + model.lambda * I
    model.output_weights = gram \ (hidden_output' * targets)
    model.classes = classes
    return model
end

function _require_fitted(model::ELMClassifier)
    isnothing(model.classes) && throw(ArgumentError("the model must be fitted before prediction"))
    return nothing
end

"""Return one score per class for every row in `X`."""
function predict_scores(model::ELMClassifier{T}, X::AbstractMatrix) where {T}
    _require_fitted(model)
    size(X, 2) == model.input_size || throw(DimensionMismatch(
        "X has $(size(X, 2)) features; the fitted model expects $(model.input_size)",
    ))
    features = _checked_features(T, X)
    hidden_output = model.activation.(features * model.input_weights .+ model.biases')
    return hidden_output * model.output_weights
end

"""Predict the original class label for every row in `X`."""
function predict(model::ELMClassifier, X::AbstractMatrix)
    scores = predict_scores(model, X)
    indices = [argmax(@view scores[row, :]) for row in axes(scores, 1)]
    return model.classes[indices]
end

function _binary_class_indices(model::ELMClassifier, positive_label)
    _require_fitted(model)
    length(model.classes) == 2 || throw(ArgumentError(
        "binary thresholding requires a model fitted with exactly two classes",
    ))
    positive_index = isnothing(positive_label) ?
                     (1 in model.classes ? findfirst(==(1), model.classes) : 2) :
                     findfirst(==(positive_label), model.classes)
    isnothing(positive_index) && throw(ArgumentError(
        "positive_label is absent from the fitted model classes",
    ))
    negative_index = only(index for index in eachindex(model.classes) if index != positive_index)
    return positive_index, negative_index
end

"""
    binary_decision_scores(model, X; positive_label=nothing)

Return the positive-minus-negative ELM score for each row of a binary model.
The ordinary `predict` decision boundary corresponds to a score of zero.
"""
function binary_decision_scores(
    model::ELMClassifier,
    X::AbstractMatrix;
    positive_label=nothing,
)
    positive_index, negative_index = _binary_class_indices(model, positive_label)
    scores = predict_scores(model, X)
    return scores[:, positive_index] .- scores[:, negative_index]
end

"""
    predict_with_threshold(model, X, threshold; positive_label=nothing)

Classify a binary ELM with a threshold applied to its positive-minus-negative
decision score. Scores greater than or equal to `threshold` are positive.
"""
function predict_with_threshold(
    model::ELMClassifier,
    X::AbstractMatrix,
    threshold::Real;
    positive_label=nothing,
)
    isfinite(threshold) || throw(ArgumentError("threshold must be finite"))
    positive_index, negative_index = _binary_class_indices(model, positive_label)
    decision_scores = binary_decision_scores(model, X; positive_label=model.classes[positive_index])
    positive = model.classes[positive_index]
    negative = model.classes[negative_index]
    return [score >= threshold ? positive : negative for score in decision_scores]
end

"""
    optimal_f1_threshold(scores, y; positive_label=1, minimum_recall=nothing,
                         minimum_specificity=nothing)

Find the decision-score threshold that maximizes binary F1. All equal scores
are assigned to the same side of the threshold, making the returned rule
deterministic and suitable for a held-out validation set. When
`minimum_recall` and/or `minimum_specificity` are supplied, only thresholds
meeting those validation targets are eligible.
"""
function optimal_f1_threshold(
    scores::AbstractVector{<:Real},
    y::AbstractVector;
    positive_label=1,
    minimum_recall::Union{Nothing,Real}=nothing,
    minimum_specificity::Union{Nothing,Real}=nothing,
)
    length(scores) == length(y) || throw(DimensionMismatch(
        "scores and labels must have the same length",
    ))
    isempty(scores) && throw(ArgumentError("scores must not be empty"))
    all(isfinite, scores) || throw(ArgumentError("scores contain NaN or infinite values"))
    positive_label in y || throw(ArgumentError("positive_label is absent from y"))
    classes = unique(y)
    length(classes) == 2 || throw(ArgumentError(
        "threshold calibration requires exactly two labels",
    ))
    !isnothing(minimum_recall) && !(0 <= minimum_recall <= 1) && throw(ArgumentError(
        "minimum_recall must be between zero and one",
    ))
    !isnothing(minimum_specificity) && !(0 <= minimum_specificity <= 1) && throw(ArgumentError(
        "minimum_specificity must be between zero and one",
    ))

    positive_total = count(==(positive_label), y)
    negative_total = length(y) - positive_total
    order = sortperm(scores; rev=true)
    true_positive = 0
    false_positive = 0
    best_f1 = -1.0
    best_threshold = Float64(scores[first(order)])
    best_tp = 0
    best_fp = 0
    index = firstindex(order)

    while index <= lastindex(order)
        threshold = scores[order[index]]
        last_equal = index
        while last_equal <= lastindex(order) && scores[order[last_equal]] == threshold
            y[order[last_equal]] == positive_label ? (true_positive += 1) : (false_positive += 1)
            last_equal += 1
        end
        false_negative = positive_total - true_positive
        current_recall = _safe_ratio(true_positive, true_positive + false_negative)
        current_specificity = _safe_ratio(negative_total - false_positive, negative_total)
        current_f1 = _safe_ratio(
            2 * true_positive,
            2 * true_positive + false_positive + false_negative,
        )
        eligible_recall = isnothing(minimum_recall) || current_recall >= minimum_recall
        eligible_specificity = isnothing(minimum_specificity) ||
                               current_specificity >= minimum_specificity
        if eligible_recall && eligible_specificity && current_f1 > best_f1
            best_f1 = current_f1
            best_threshold = Float64(threshold)
            best_tp = true_positive
            best_fp = false_positive
        end
        index = last_equal
    end

    best_f1 >= 0 || throw(ArgumentError(
        "no threshold meets the requested recall/specificity constraints",
    ))
    best_fn = positive_total - best_tp
    predicted_negative = length(y) - best_tp - best_fp - best_fn
    return (
        threshold=best_threshold,
        f1=best_f1,
        precision=_safe_ratio(best_tp, best_tp + best_fp),
        recall=_safe_ratio(best_tp, best_tp + best_fn),
        specificity=_safe_ratio(predicted_negative, predicted_negative + best_fp),
        accuracy=(best_tp + predicted_negative) / length(y),
        true_negative=predicted_negative,
        false_positive=best_fp,
        false_negative=best_fn,
        true_positive=best_tp,
    )
end

end
