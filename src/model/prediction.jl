function _require_fitted(model::ELMClassifier)
    isnothing(model.classes) && throw(ArgumentError("the model must be fitted before prediction"))
    return nothing
end

function predict_scores(model::ELMClassifier{T}, X::AbstractMatrix) where {T}
    _require_fitted(model)
    size(X, 2) == model.input_size || throw(DimensionMismatch(
        "X has $(size(X, 2)) features; the fitted model expects $(model.input_size)",
    ))
    features = _checked_features(T, X)
    hidden_output = model.activation.(features * model.input_weights .+ model.biases')
    return hidden_output * model.output_weights
end

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

function binary_decision_scores(
    model::ELMClassifier,
    X::AbstractMatrix;
    positive_label=nothing,
)
    positive_index, negative_index = _binary_class_indices(model, positive_label)
    scores = predict_scores(model, X)
    return scores[:, positive_index] .- scores[:, negative_index]
end

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
