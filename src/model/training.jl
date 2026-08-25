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
