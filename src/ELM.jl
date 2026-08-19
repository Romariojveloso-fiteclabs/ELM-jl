module ELM

using LinearAlgebra
using Random
import Base: precision

include("activations.jl")
include("preprocessing.jl")
include("metrics.jl")

export ELMClassifier, fit!, predict, predict_scores,
       abs_activation, tanh_activation,
       load_dataset, normalize_features, NormalizationParams,
       encode_labels, train_test_split,
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

end
