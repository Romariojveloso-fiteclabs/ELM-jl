using Serialization

const IOT23_NUMERIC_NAMES = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
]
const IOT23_CATEGORICAL_NAMES = ["proto", "service", "conn_state"]
const IOT23_UNKNOWN_CATEGORY = "__UNKNOWN__"
const IOT23_EXPECTED_HEADER = join([
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents   label   detailed-label",
], ',')

mutable struct _RunningStat
    count::Int
    mean::Float64
    m2::Float64
end

_RunningStat() = _RunningStat(0, 0.0, 0.0)

function _observe!(stat::_RunningStat, value::Float64)
    stat.count += 1
    delta = value - stat.mean
    stat.mean += delta / stat.count
    stat.m2 += delta * (value - stat.mean)
    return
end

"""
Parameters learned exclusively from IoT-23 training scenarios.

Numerical values are optionally transformed with `log1p`, mean-imputed, and
z-score normalized. Every numerical feature has an accompanying missing-value
indicator. Categorical vocabularies include a reserved unknown column.
"""
struct IoT23Preprocessor{T<:AbstractFloat}
    numeric_names::Vector{String}
    numeric_means::Vector{Float64}
    numeric_scales::Vector{Float64}
    numeric_valid_counts::Vector{Int}
    categorical_names::Vector{String}
    categorical_values::Vector{Vector{String}}
    categorical_maps::Vector{Dict{String,Int}}
    feature_names::Vector{String}
    log_transform::Bool
    fitted_scenarios::Vector{String}
end

"""Exact per-scenario/per-class quotas for reproducible streaming sampling."""
struct IoT23SamplingPlan
    scenarios::Vector{String}
    available::Dict{Tuple{String,Int8},Int}
    selected::Dict{Tuple{String,Int8},Int}
    seed::Int
end

Base.eltype(::IoT23Preprocessor{T}) where {T} = T

function _iot23_header(input, path)
    eof(input) && throw(ArgumentError("empty IoT-23 file: $path"))
    header = readline(input)
    endswith(header, '\r') && (header = chop(header))
    header == IOT23_EXPECTED_HEADER || throw(ArgumentError(
        "unexpected IoT-23 schema in $path",
    ))
    return
end

function _iot23_row(line::String)
    empty = SubString(line, 1, 0)
    proto = service = duration = orig_bytes = resp_bytes = conn_state = empty
    orig_pkts = orig_ip_bytes = resp_pkts = resp_ip_bytes = empty
    field = 1
    field_start = 1
    bytes = codeunits(line)

    for position in eachindex(bytes)
        bytes[position] == UInt8(',') || continue
        value = SubString(line, field_start, position - 1)
        field == 7 && (proto = value)
        field == 8 && (service = value)
        field == 9 && (duration = value)
        field == 10 && (orig_bytes = value)
        field == 11 && (resp_bytes = value)
        field == 12 && (conn_state = value)
        field == 17 && (orig_pkts = value)
        field == 18 && (orig_ip_bytes = value)
        field == 19 && (resp_pkts = value)
        field == 20 && (resp_ip_bytes = value)
        field += 1
        field_start = position + 1
    end

    field == 21 || return nothing
    line_end = lastindex(bytes)
    line_end >= field_start && bytes[line_end] == UInt8('\r') && (line_end -= 1)
    combined_label = SubString(line, field_start, line_end)
    return (
        numeric=(duration, orig_bytes, resp_bytes, orig_pkts,
                 resp_pkts, orig_ip_bytes, resp_ip_bytes),
        categorical=(proto, service, conn_state),
        combined_label=combined_label,
    )
end

function _iot23_numeric(value::AbstractString, use_log::Bool)
    text = strip(value)
    (isempty(text) || text == "-" || text == "?") && return false, 0.0
    parsed = tryparse(Float64, text)
    (isnothing(parsed) || !isfinite(parsed) || parsed < 0) && return false, 0.0
    return true, use_log ? log1p(parsed) : parsed
end

function _iot23_binary_label(combined::AbstractString)
    parts = split(strip(combined); limit=3)
    length(parts) == 3 || return nothing
    label = parts[2]
    (label == "benign" || label == "Benign") && return Int8(0)
    (label == "malicious" || label == "Malicious") && return Int8(1)
    return nothing
end

"""
    fit_iot23_preprocessor(train_paths; datatype=Float32, log_transform=true)

Learn numerical statistics and categorical vocabularies by streaming only the
given training scenarios. No rows or parameters from a test scenario are used.
"""
function fit_iot23_preprocessor(
    train_paths::AbstractVector{<:AbstractString};
    datatype::Type{T}=Float32,
    log_transform::Bool=true,
    progress::Bool=true,
) where {T<:AbstractFloat}
    isempty(train_paths) && throw(ArgumentError("at least one training scenario is required"))
    stats = [_RunningStat() for _ in IOT23_NUMERIC_NAMES]
    vocabularies = [Set{String}() for _ in IOT23_CATEGORICAL_NAMES]

    for (index, path) in enumerate(train_paths)
        isfile(path) || throw(ArgumentError("training scenario does not exist: $path"))
        progress && println("[$index/$(length(train_paths))] ajustando com $(basename(path))")
        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                for feature in eachindex(stats)
                    valid, value = _iot23_numeric(row.numeric[feature], log_transform)
                    valid && _observe!(stats[feature], value)
                end
                for feature in eachindex(vocabularies)
                    value = String(strip(row.categorical[feature]))
                    push!(vocabularies[feature], value)
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

    feature_names = copy(IOT23_NUMERIC_NAMES)
    append!(feature_names, ["$(name)_missing" for name in IOT23_NUMERIC_NAMES])
    for (name, values) in zip(IOT23_CATEGORICAL_NAMES, categorical_values)
        append!(feature_names, ["$(name)=$(value)" for value in values])
        push!(feature_names, "$(name)=$(IOT23_UNKNOWN_CATEGORY)")
    end

    return IoT23Preprocessor{T}(
        copy(IOT23_NUMERIC_NAMES), means, scales, [stat.count for stat in stats],
        copy(IOT23_CATEGORICAL_NAMES), categorical_values, categorical_maps,
        feature_names, log_transform, basename.(String.(train_paths)),
    )
end

function _transform_iot23_row!(X, y, output_row, row, preprocessor::IoT23Preprocessor{T}) where {T}
    numeric_count = length(preprocessor.numeric_names)
    for feature in 1:numeric_count
        valid, value = _iot23_numeric(row.numeric[feature], preprocessor.log_transform)
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

"""
    foreach_iot23_batch(f, paths, preprocessor; batch_size=100_000)

Transform IoT-23 scenarios in bounded-memory batches and call
`f(Xbatch, ybatch, scenario)` for each batch. Views passed to `f` are reused and
must not be retained after the callback returns.
"""
function foreach_iot23_batch(
    f::F,
    paths::AbstractVector{<:AbstractString},
    preprocessor::IoT23Preprocessor{T};
    batch_size::Integer=100_000,
) where {F,T}
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    feature_count = length(preprocessor.feature_names)
    X = zeros(T, batch_size, feature_count)
    y = Vector{Int8}(undef, batch_size)
    total_rows = 0
    benign = 0
    malicious = 0

    for path in paths
        scenario = basename(path)
        filled = 0
        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                filled += 1
                _transform_iot23_row!(X, y, filled, row, preprocessor) || throw(ArgumentError(
                    "invalid binary label in $path",
                ))
                total_rows += 1
                y[filled] == 0 ? (benign += 1) : (malicious += 1)
                if filled == batch_size
                    f(@view(X[1:filled, :]), @view(y[1:filled]), scenario)
                    fill!(X, zero(T))
                    filled = 0
                end
            end
        end
        if filled > 0
            f(@view(X[1:filled, :]), @view(y[1:filled]), scenario)
            fill!(X, zero(T))
        end
    end
    return (rows=total_rows, benign=benign, malicious=malicious)
end

function _equalized_quotas(capacities::Vector{Int}, target::Int)
    0 <= target <= sum(capacities) || throw(ArgumentError(
        "sampling target must be between zero and the available population",
    ))
    quotas = zeros(Int, length(capacities))
    remaining = target
    while remaining > 0
        active = [index for index in eachindex(capacities) if quotas[index] < capacities[index]]
        isempty(active) && error("unable to distribute the requested sampling quota")
        base, extra = divrem(remaining, length(active))
        distributed = 0
        for (position, index) in enumerate(active)
            requested = base + (position <= extra ? 1 : 0)
            requested = max(requested, 1)
            amount = min(requested, capacities[index] - quotas[index], remaining - distributed)
            quotas[index] += amount
            distributed += amount
            distributed == remaining && break
        end
        remaining -= distributed
    end
    return quotas
end

"""
    balanced_iot23_sampling_plan(scenarios, benign_counts, malicious_counts;
                                 target_per_class=500_000, seed=42)

Create exact class-balanced quotas. Within each class, scenarios receive
approximately equal quotas; unused capacity from small scenarios is
redistributed among the remaining scenarios.
"""
function balanced_iot23_sampling_plan(
    scenarios::AbstractVector{<:AbstractString},
    benign_counts::AbstractVector{<:Integer},
    malicious_counts::AbstractVector{<:Integer};
    target_per_class::Integer=500_000,
    seed::Integer=42,
)
    length(scenarios) == length(benign_counts) == length(malicious_counts) ||
        throw(DimensionMismatch("scenario and count vectors must have the same length"))
    length(unique(scenarios)) == length(scenarios) || throw(ArgumentError(
        "scenario names must be unique",
    ))
    target_per_class > 0 || throw(ArgumentError("target_per_class must be positive"))
    benign = Int.(benign_counts)
    malicious = Int.(malicious_counts)
    any(<(0), benign) && throw(ArgumentError("benign counts must be non-negative"))
    any(<(0), malicious) && throw(ArgumentError("malicious counts must be non-negative"))

    benign_quotas = _equalized_quotas(benign, Int(target_per_class))
    malicious_quotas = _equalized_quotas(malicious, Int(target_per_class))
    names = String.(scenarios)
    available = Dict{Tuple{String,Int8},Int}()
    selected = Dict{Tuple{String,Int8},Int}()
    for index in eachindex(names)
        available[(names[index], Int8(0))] = benign[index]
        available[(names[index], Int8(1))] = malicious[index]
        selected[(names[index], Int8(0))] = benign_quotas[index]
        selected[(names[index], Int8(1))] = malicious_quotas[index]
    end
    return IoT23SamplingPlan(names, available, selected, Int(seed))
end

"""
    foreach_sampled_iot23_batch(f, paths, preprocessor, plan; batch_size=25_000)

Uniformly sample without replacement inside each scenario/class quota while
streaming the raw files. The exact planned number of rows is emitted.
"""
function foreach_sampled_iot23_batch(
    f::F,
    paths::AbstractVector{<:AbstractString},
    preprocessor::IoT23Preprocessor{T},
    plan::IoT23SamplingPlan;
    batch_size::Integer=25_000,
) where {F,T}
    batch_size > 0 || throw(ArgumentError("batch_size must be positive"))
    basename.(String.(paths)) == plan.scenarios || throw(ArgumentError(
        "paths must follow the same scenario order as the sampling plan",
    ))
    rng = MersenneTwister(plan.seed)
    feature_count = length(preprocessor.feature_names)
    X = zeros(T, batch_size, feature_count)
    y = Vector{Int8}(undef, batch_size)
    sampled_counts = zeros(Int, 2)
    scanned_rows = 0

    for path in paths
        scenario = basename(path)
        available_remaining = [
            plan.available[(scenario, Int8(0))],
            plan.available[(scenario, Int8(1))],
        ]
        target_remaining = [
            plan.selected[(scenario, Int8(0))],
            plan.selected[(scenario, Int8(1))],
        ]
        filled = 0

        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                label = _iot23_binary_label(row.combined_label)
                isnothing(label) && throw(ArgumentError("invalid binary label in $path"))
                class_index = Int(label) + 1
                population = available_remaining[class_index]
                population > 0 || throw(ArgumentError(
                    "profile count is smaller than raw rows for $scenario class $label",
                ))
                target = target_remaining[class_index]
                accept = target > 0 && rand(rng, 1:population) <= target
                available_remaining[class_index] -= 1
                scanned_rows += 1
                accept || continue

                target_remaining[class_index] -= 1
                filled += 1
                _transform_iot23_row!(X, y, filled, row, preprocessor) || error(
                    "failed to transform a sampled row",
                )
                sampled_counts[class_index] += 1
                if filled == batch_size
                    f(@view(X[1:filled, :]), @view(y[1:filled]), scenario)
                    fill!(X, zero(T))
                    filled = 0
                end
            end
        end

        all(iszero, available_remaining) || throw(ArgumentError(
            "profile count is larger than raw rows for $scenario",
        ))
        all(iszero, target_remaining) || error("sampling quota was not fulfilled for $scenario")
        if filled > 0
            f(@view(X[1:filled, :]), @view(y[1:filled]), scenario)
            fill!(X, zero(T))
        end
    end
    return (
        scanned_rows=scanned_rows,
        sampled_rows=sum(sampled_counts),
        benign=sampled_counts[1],
        malicious=sampled_counts[2],
    )
end

function save_iot23_preprocessor(path::AbstractString, preprocessor::IoT23Preprocessor)
    mkpath(dirname(path))
    open(path, "w") do output
        serialize(output, preprocessor)
    end
    return path
end

function load_iot23_preprocessor(path::AbstractString)
    open(path, "r") do input
        value = deserialize(input)
        value isa IoT23Preprocessor || throw(ArgumentError("invalid IoT-23 preprocessor file"))
        return value
    end
end
