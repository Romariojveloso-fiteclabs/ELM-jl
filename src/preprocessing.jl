using DelimitedFiles
using Random

"""Parameters learned by `normalize_features` and reusable on new data."""
struct NormalizationParams{T<:AbstractFloat}
    method::Symbol
    offset::Vector{T}
    scale::Vector{T}
end

_feature_type(X::AbstractMatrix) = eltype(X) <: AbstractFloat ? eltype(X) : Float64

function _as_feature_matrix(::Type{T}, X::AbstractMatrix) where {T<:AbstractFloat}
    converted = try
        Matrix{T}(X)
    catch error
        throw(ArgumentError("features must be numeric and convertible to $T: $error"))
    end
    all(isfinite, converted) || throw(ArgumentError("features contain NaN or infinite values"))
    return converted
end

"""
    load_dataset(path; target=:last, delimiter=',', header=true, datatype=Float64)

Load a delimited dataset. `target` may be a one-based column index, header
name, or `:last`. Features must be numeric; labels retain their parsed type.
"""
function load_dataset(
    path::AbstractString;
    target=:last,
    delimiter::Char=',',
    header::Bool=true,
    datatype::Type{T}=Float64,
) where {T<:AbstractFloat}
    isfile(path) || throw(ArgumentError("dataset file does not exist: $path"))
    if header
        raw, raw_header = readdlm(path, delimiter, Any; header=true)
        names = string.(vec(raw_header))
    else
        raw = readdlm(path, delimiter, Any)
        names = String[]
    end
    raw = ndims(raw) == 1 ? reshape(raw, 1, :) : raw
    size(raw, 1) > 0 || throw(ArgumentError("dataset has no data rows"))
    size(raw, 2) >= 2 || throw(ArgumentError("dataset must contain features and a target"))

    target_index = if target === :last
        size(raw, 2)
    elseif target isa Integer
        Int(target)
    elseif target isa Union{AbstractString,Symbol}
        header || throw(ArgumentError("target by name requires header=true"))
        index = findfirst(==(string(target)), names)
        isnothing(index) && throw(ArgumentError("target column '$(target)' was not found"))
        index
    else
        throw(ArgumentError("target must be :last, a column index, or a header name"))
    end
    1 <= target_index <= size(raw, 2) || throw(BoundsError(raw, (:, target_index)))

    feature_indices = [column for column in axes(raw, 2) if column != target_index]
    X = Matrix{T}(undef, size(raw, 1), length(feature_indices))
    for (output_column, input_column) in enumerate(feature_indices), row in axes(raw, 1)
        value = raw[row, input_column]
        X[row, output_column] = try
            value isa Number ? T(value) : parse(T, strip(string(value)))
        catch
            column_name = header ? names[input_column] : string(input_column)
            throw(ArgumentError(
                "non-numeric feature at row $row, column '$column_name': $(repr(value))",
            ))
        end
    end
    all(isfinite, X) || throw(ArgumentError("dataset contains NaN or infinite feature values"))
    y = [value isa AbstractString ? strip(value) : value for value in raw[:, target_index]]
    return X, y
end

"""Normalize columns with `:zscore` or `:minmax` statistics."""
function normalize_features(
    X::AbstractMatrix;
    method::Symbol=:zscore,
    return_params::Bool=false,
)
    method in (:zscore, :minmax) || throw(ArgumentError(
        "normalization method must be :zscore or :minmax",
    ))
    isempty(X) && throw(ArgumentError("X must not be empty"))
    T = _feature_type(X)
    features = _as_feature_matrix(T, X)
    offset = Vector{T}(undef, size(features, 2))
    scale = Vector{T}(undef, size(features, 2))
    for column in axes(features, 2)
        values = @view features[:, column]
        if method === :zscore
            offset[column] = sum(values) / length(values)
            centered = values .- offset[column]
            scale[column] = sqrt(sum(abs2, centered) / length(values))
        else
            minimum_value, maximum_value = extrema(values)
            offset[column] = minimum_value
            scale[column] = maximum_value - minimum_value
        end
        iszero(scale[column]) && (scale[column] = one(T))
    end
    params = NormalizationParams(method, offset, scale)
    normalized = normalize_features(features, params)
    return return_params ? (normalized, params) : normalized
end

"""Apply normalization parameters learned from training data."""
function normalize_features(X::AbstractMatrix, params::NormalizationParams{T}) where {T}
    size(X, 2) == length(params.offset) || throw(DimensionMismatch(
        "X has $(size(X, 2)) features but normalization expects $(length(params.offset))",
    ))
    features = _as_feature_matrix(T, X)
    return (features .- params.offset') ./ params.scale'
end

"""Return `(Y, classes)`, with labels represented as one-hot columns."""
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

"""Split rows into reproducible training and test partitions."""
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
