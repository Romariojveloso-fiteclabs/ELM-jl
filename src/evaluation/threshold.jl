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
