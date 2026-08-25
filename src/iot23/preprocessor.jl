function fit_iot23_preprocessor(
    train_paths::AbstractVector{<:AbstractString};
    datatype::Type{T}=Float32,
    log_transform::Bool=true,
    representation::Symbol=:baseline,
    progress::Bool=true,
) where {T<:AbstractFloat}
    representation in (:baseline, :light, :behavioral) || throw(ArgumentError(
        "invalid representation: $representation. Expected :baseline, :light, or :behavioral",
    ))
    numeric_names = representation == :baseline ? copy(IOT23_NUMERIC_NAMES) :
                    representation == :light ? copy(IOT23_LIGHT_NUMERIC_NAMES) :
                    copy(IOT23_BEHAVIORAL_NUMERIC_NAMES)

    categorical_names = representation == :baseline ? copy(IOT23_CATEGORICAL_NAMES) :
                        copy(IOT23_LIGHT_CATEGORICAL_NAMES)

    stats = [_RunningStat() for _ in numeric_names]
    vocabularies = [Set{String}() for _ in categorical_names]
    tracker = OnlineScenarioTracker()

    for (index, path) in enumerate(train_paths)
        isfile(path) || throw(ArgumentError("training scenario does not exist: $path"))
        progress && println("[$index/$(length(train_paths))] ajustando com $(basename(path))")
        reset!(tracker)
        row_seq = 0.0
        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                row_seq += 1.0
                vals = _extract_raw_numerics(row, representation, tracker, row_seq, log_transform)
                for feature in eachindex(stats)
                    valid, val = vals[feature]
                    valid && _observe!(stats[feature], val)
                end
                for feature in eachindex(vocabularies)
                    val = String(strip(row.categorical[feature]))
                    push!(vocabularies[feature], val)
                end
            end
        end
    end

    any(stat -> stat.count == 0, stats) && throw(ArgumentError(
        "one or more numerical features have no valid training values",
    ))
    means = [stat.mean for stat in stats]
    scales = [stat.count > 0 ? sqrt(stat.m2 / stat.count) : 1.0 for stat in stats]
    scales = [iszero(scale) ? 1.0 : scale for scale in scales]
    categorical_values = [sort!(collect(values)) for values in vocabularies]
    categorical_maps = [
        Dict(value => index for (index, value) in enumerate(values))
        for values in categorical_values
    ]

    feature_names = copy(numeric_names)
    append!(feature_names, ["$(name)_missing" for name in numeric_names])
    for (name, values) in zip(categorical_names, categorical_values)
        append!(feature_names, ["$(name)=$(value)" for value in values])
        push!(feature_names, "$(name)=$(IOT23_UNKNOWN_CATEGORY)")
    end

    return IoT23Preprocessor{T}(
        numeric_names, means, scales, [stat.count for stat in stats],
        categorical_names, categorical_values, categorical_maps,
        feature_names, log_transform, basename.(String.(train_paths)),
    )
end

function fit_all_iot23_preprocessors(
    train_paths::AbstractVector{<:AbstractString};
    datatype::Type{T}=Float32,
    log_transform::Bool=true,
    progress::Bool=true,
) where {T<:AbstractFloat}
    isempty(train_paths) && throw(ArgumentError("at least one training scenario is required"))

    numeric_names_all = copy(IOT23_BEHAVIORAL_NUMERIC_NAMES)
    categorical_names_all = copy(IOT23_BEHAVIORAL_CATEGORICAL_NAMES)

    stats = [_RunningStat() for _ in numeric_names_all]
    vocabularies = [Set{String}() for _ in categorical_names_all]
    tracker = OnlineScenarioTracker()

    for (index, path) in enumerate(train_paths)
        isfile(path) || throw(ArgumentError("training scenario does not exist: $path"))
        progress && println("[$index/$(length(train_paths))] ajustando estatísticas (todas representações) com $(basename(path))")
        reset!(tracker)
        row_seq = 0.0
        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                row_seq += 1.0
                vals = _extract_raw_numerics(row, :behavioral, tracker, row_seq, log_transform)
                for feature in eachindex(stats)
                    valid, val = vals[feature]
                    valid && _observe!(stats[feature], val)
                end
                for feature in eachindex(vocabularies)
                    val = String(strip(row.categorical[feature]))
                    push!(vocabularies[feature], val)
                end
            end
        end
    end

    any(stat -> stat.count == 0, stats) && throw(ArgumentError(
        "one or more numerical features have no valid training values",
    ))
    all_means = [stat.mean for stat in stats]
    all_scales = [stat.count > 0 ? sqrt(stat.m2 / stat.count) : 1.0 for stat in stats]
    all_scales = [iszero(scale) ? 1.0 : scale for scale in all_scales]
    all_valid_counts = [stat.count for stat in stats]
    all_categorical_values = [sort!(collect(values)) for values in vocabularies]
    all_categorical_maps = [
        Dict(value => index for (index, value) in enumerate(values))
        for values in all_categorical_values
    ]

    scenarios = basename.(String.(train_paths))
    results = Dict{Symbol, IoT23Preprocessor{T}}()

    for rep in (:baseline, :light, :behavioral)
        num_names = rep == :baseline ? copy(IOT23_NUMERIC_NAMES) :
                    rep == :light ? copy(IOT23_LIGHT_NUMERIC_NAMES) :
                    copy(IOT23_BEHAVIORAL_NUMERIC_NAMES)

        cat_names = rep == :baseline ? copy(IOT23_CATEGORICAL_NAMES) :
                    copy(IOT23_LIGHT_CATEGORICAL_NAMES)

        num_indices = [findfirst(==(name), numeric_names_all) for name in num_names]
        cat_indices = [findfirst(==(name), categorical_names_all) for name in cat_names]

        means = all_means[num_indices]
        scales = all_scales[num_indices]
        counts = all_valid_counts[num_indices]
        cat_vals = all_categorical_values[cat_indices]
        cat_maps = all_categorical_maps[cat_indices]

        feature_names = copy(num_names)
        append!(feature_names, ["$(name)_missing" for name in num_names])
        for (name, values) in zip(cat_names, cat_vals)
            append!(feature_names, ["$(name)=$(value)" for value in values])
            push!(feature_names, "$(name)=$(IOT23_UNKNOWN_CATEGORY)")
        end

        results[rep] = IoT23Preprocessor{T}(
            num_names, means, scales, counts,
            cat_names, cat_vals, cat_maps,
            feature_names, log_transform, scenarios,
        )
    end

    return results
end

function _transform_iot23_row!(
    X, y, output_row, row, preprocessor::IoT23Preprocessor{T},
    tracker::Union{Nothing,OnlineScenarioTracker}=nothing, row_seq::Float64=0.0,
) where {T}
    raw_numerics = _extract_raw_numerics(
        row, preprocessor.representation, something(tracker, OnlineScenarioTracker()),
        row_seq, preprocessor.log_transform,
    )
    numeric_count = length(preprocessor.numeric_names)

    for feature in 1:numeric_count
        valid, value = raw_numerics[feature]
        X[output_row, feature] = valid ? T(
            (value - preprocessor.numeric_means[feature]) /
            preprocessor.numeric_scales[feature]
        ) : zero(T)
        X[output_row, numeric_count + feature] = valid ? zero(T) : one(T)
    end

    output_column = 2 * numeric_count + 1
    for feature in eachindex(preprocessor.categorical_names)
        value = strip(row.categorical[feature])
        values = preprocessor.categorical_values[feature]
        index = get(preprocessor.categorical_maps[feature], value, length(values) + 1)
        X[output_row, output_column + index - 1] = one(T)
        output_column += length(values) + 1
    end
    label = _iot23_binary_label(row.combined_label)
    isnothing(label) && return false
    y[output_row] = label
    return true
end
