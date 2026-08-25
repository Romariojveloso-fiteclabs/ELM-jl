using Random

function encode_labels(
    y::AbstractVector;
    classes=unique(y),
    datatype::Type{T}=Float64,
) where {T<:AbstractFloat}
    isempty(y) && throw(ArgumentError("y must not be empty"))
    class_values = collect(classes)
    isempty(class_values) && throw(ArgumentError("classes must not be empty"))
    length(unique(class_values)) == length(class_values) || throw(ArgumentError(
        "classes must not contain duplicates",
    ))
    mapping = Dict(class => index for (index, class) in enumerate(class_values))
    encoded = zeros(T, length(y), length(class_values))
    for (row, label) in enumerate(y)
        haskey(mapping, label) || throw(ArgumentError("unknown label: $(repr(label))"))
        encoded[row, mapping[label]] = one(T)
    end
    return encoded, class_values
end

function train_test_split(
    X::AbstractMatrix,
    y::AbstractVector;
    test_size::Real=0.2,
    seed::Integer=42,
    shuffle::Bool=true,
)
    size(X, 1) == length(y) || throw(DimensionMismatch(
        "X has $(size(X, 1)) rows but y has $(length(y)) labels",
    ))
    length(y) >= 2 || throw(ArgumentError("at least two samples are required"))
    0 < test_size < 1 || throw(ArgumentError("test_size must be between 0 and 1"))
    indices = collect(eachindex(y))
    shuffle && Random.shuffle!(MersenneTwister(seed), indices)
    test_count = clamp(round(Int, length(y) * test_size), 1, length(y) - 1)
    test_indices = indices[1:test_count]
    train_indices = indices[(test_count + 1):end]
    return X[train_indices, :], X[test_indices, :], y[train_indices], y[test_indices]
end
