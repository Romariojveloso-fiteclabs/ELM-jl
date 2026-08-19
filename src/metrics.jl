function _validate_labels(y_true::AbstractVector, y_pred::AbstractVector)
    length(y_true) == length(y_pred) || throw(DimensionMismatch(
        "y_true and y_pred must have the same length",
    ))
    isempty(y_true) && throw(ArgumentError("label vectors must not be empty"))
    return unique(vcat(y_true, y_pred))
end

_safe_ratio(numerator, denominator) = iszero(denominator) ? 0.0 : numerator / denominator

function _score(metric::Symbol, tp, fp, fn)
    metric === :precision && return _safe_ratio(tp, tp + fp)
    metric === :recall && return _safe_ratio(tp, tp + fn)
    p = _safe_ratio(tp, tp + fp)
    r = _safe_ratio(tp, tp + fn)
    return _safe_ratio(2 * p * r, p + r)
end

function _class_counts(y_true, y_pred, label)
    tp = count(i -> y_true[i] == label && y_pred[i] == label, eachindex(y_true))
    fp = count(i -> y_true[i] != label && y_pred[i] == label, eachindex(y_true))
    fn = count(i -> y_true[i] == label && y_pred[i] != label, eachindex(y_true))
    return tp, fp, fn, count(==(label), y_true)
end

function _positive_label(classes, positive_label)
    if !isnothing(positive_label)
        positive_label in classes || throw(ArgumentError("positive_label is absent from the labels"))
        return positive_label
    elseif 1 in classes
        return first(class for class in classes if class == 1)
    end
    return last(classes)
end

function _average_metric(metric::Symbol, y_true, y_pred; average::Symbol=:auto, positive_label=nothing)
    classes = _validate_labels(y_true, y_pred)
    selected = average === :auto ? (length(classes) <= 2 ? :binary : :macro) : average
    selected in (:binary, :macro, :micro, :weighted) || throw(ArgumentError(
        "average must be :auto, :binary, :macro, :micro, or :weighted",
    ))
    if selected === :binary
        length(classes) <= 2 || throw(ArgumentError("binary averaging requires at most two classes"))
        tp, fp, fn, _ = _class_counts(y_true, y_pred, _positive_label(classes, positive_label))
        return _score(metric, tp, fp, fn)
    elseif selected === :micro
        counts = [_class_counts(y_true, y_pred, label) for label in classes]
        tp = sum(first, counts)
        fp = sum(c[2] for c in counts)
        fn = sum(c[3] for c in counts)
        return _score(metric, tp, fp, fn)
    end
    values = Float64[]
    supports = Int[]
    for label in classes
        tp, fp, fn, support = _class_counts(y_true, y_pred, label)
        push!(values, _score(metric, tp, fp, fn))
        push!(supports, support)
    end
    return selected === :macro ? sum(values) / length(values) :
           sum(values .* supports) / sum(supports)
end

"""Fraction of exactly correct predictions."""
function accuracy(y_true::AbstractVector, y_pred::AbstractVector)
    _validate_labels(y_true, y_pred)
    return count(i -> y_true[i] == y_pred[i], eachindex(y_true)) / length(y_true)
end

"""Precision with automatic binary/macro averaging."""
precision(y_true::AbstractVector, y_pred::AbstractVector; average::Symbol=:auto, positive_label=nothing) =
    _average_metric(:precision, y_true, y_pred; average, positive_label)

"""Recall with automatic binary/macro averaging."""
recall(y_true::AbstractVector, y_pred::AbstractVector; average::Symbol=:auto, positive_label=nothing) =
    _average_metric(:recall, y_true, y_pred; average, positive_label)

"""Harmonic mean of precision and recall."""
function f1_score(y_true::AbstractVector, y_pred::AbstractVector; average::Symbol=:auto, positive_label=nothing)
    return _average_metric(:f1, y_true, y_pred; average, positive_label)
end
