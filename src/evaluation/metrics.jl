import Base: precision

_safe_ratio(numerator::Real, denominator::Real) = denominator > 0 ? Float64(numerator) / Float64(denominator) : 0.0

function accuracy(y_true::AbstractVector, y_pred::AbstractVector)
    length(y_true) == length(y_pred) || throw(DimensionMismatch(
        "y_true and y_pred must have the same length",
    ))
    isempty(y_true) && throw(ArgumentError("y_true must not be empty"))
    return count(y_true .== y_pred) / length(y_true)
end

function precision(y_true::AbstractVector, y_pred::AbstractVector; positive_label=nothing)
    length(y_true) == length(y_pred) || throw(DimensionMismatch(
        "y_true and y_pred must have the same length",
    ))
    isempty(y_true) && throw(ArgumentError("y_true must not be empty"))
    pos = isnothing(positive_label) ? (1 in y_true ? 1 : first(y_true)) : positive_label
    pos in y_true || throw(ArgumentError("positive_label is absent from y_true"))
    tp = count(i -> y_true[i] == pos && y_pred[i] == pos, eachindex(y_true))
    fp = count(i -> y_true[i] != pos && y_pred[i] == pos, eachindex(y_true))
    return _safe_ratio(tp, tp + fp)
end

function recall(y_true::AbstractVector, y_pred::AbstractVector; positive_label=nothing)
    length(y_true) == length(y_pred) || throw(DimensionMismatch(
        "y_true and y_pred must have the same length",
    ))
    isempty(y_true) && throw(ArgumentError("y_true must not be empty"))
    pos = isnothing(positive_label) ? (1 in y_true ? 1 : first(y_true)) : positive_label
    pos in y_true || throw(ArgumentError("positive_label is absent from y_true"))
    tp = count(i -> y_true[i] == pos && y_pred[i] == pos, eachindex(y_true))
    fn = count(i -> y_true[i] == pos && y_pred[i] != pos, eachindex(y_true))
    return _safe_ratio(tp, tp + fn)
end

function f1_score(y_true::AbstractVector, y_pred::AbstractVector; positive_label=nothing)
    p = precision(y_true, y_pred; positive_label=positive_label)
    r = recall(y_true, y_pred; positive_label=positive_label)
    return _safe_ratio(2 * p * r, p + r)
end
