using DelimitedFiles

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

function normalize_features(X::AbstractMatrix, params::NormalizationParams{T}) where {T}
    size(X, 2) == length(params.offset) || throw(DimensionMismatch(
        "X has $(size(X, 2)) features but normalization expects $(length(params.offset))",
    ))
    features = _as_feature_matrix(T, X)
    return (features .- params.offset') ./ params.scale'
end
