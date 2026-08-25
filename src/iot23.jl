using Serialization
using Random

const IOT23_NUMERIC_NAMES = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes",
]
const IOT23_CATEGORICAL_NAMES = ["proto", "service", "conn_state"]

const IOT23_LIGHT_NUMERIC_NAMES = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes",
]
const IOT23_LIGHT_CATEGORICAL_NAMES = ["proto", "service", "conn_state", "id.resp_p"]

const IOT23_BEHAVIORAL_NUMERIC_NAMES = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts",
    "resp_pkts", "orig_ip_bytes", "resp_ip_bytes", "missed_bytes",
    "h_syn", "h_synack", "h_ack", "h_data", "h_fin", "h_rst",
    "win_flow_count", "win_unique_resp_p", "win_unique_resp_h", "win_same_port_ratio",
]
const IOT23_BEHAVIORAL_CATEGORICAL_NAMES = ["proto", "service", "conn_state", "id.resp_p"]

const IOT23_COMMON_PORTS = Set(["21", "22", "23", "53", "80", "123", "443", "1883", "6667", "8080", "8883"])
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

struct FlowRecord
    ts::Float64
    resp_h::String
    resp_p::String
end

mutable struct OnlineScenarioTracker
    max_window_sec::Float64
    max_history_len::Int
    history::Dict{String, Vector{FlowRecord}}
end

function OnlineScenarioTracker(; max_window_sec::Float64=60.0, max_history_len::Int=50)
    return OnlineScenarioTracker(max_window_sec, max_history_len, Dict{String, Vector{FlowRecord}}())
end

function reset!(tracker::OnlineScenarioTracker)
    empty!(tracker.history)
    return tracker
end

function get_temporal_features!(
    tracker::OnlineScenarioTracker,
    orig_h::AbstractString,
    resp_h::AbstractString,
    resp_p::AbstractString,
    ts::Float64,
)
    records = get!(tracker.history, String(orig_h)) do
        FlowRecord[]
    end

    cutoff = ts - tracker.max_window_sec
    first_valid = 1
    while first_valid <= length(records) && records[first_valid].ts < cutoff
        first_valid += 1
    end
    if first_valid > 1
        deleteat!(records, 1:(first_valid - 1))
    end

    if length(records) > tracker.max_history_len
        deleteat!(records, 1:(length(records) - tracker.max_history_len))
    end

    win_flow_count = Float64(length(records))
    if win_flow_count == 0.0
        win_unique_resp_p = 0.0
        win_unique_resp_h = 0.0
        win_same_port_ratio = 0.0
    else
        same_port_count = 0
        unique_p_count = 0
        unique_h_count = 0
        for i in 1:length(records)
            rec = records[i]
            rec.resp_p == resp_p && (same_port_count += 1)

            is_new_p = true
            for j in 1:(i - 1)
                if records[j].resp_p == rec.resp_p
                    is_new_p = false
                    break
                end
            end
            is_new_p && (unique_p_count += 1)

            is_new_h = true
            for j in 1:(i - 1)
                if records[j].resp_h == rec.resp_h
                    is_new_h = false
                    break
                end
            end
            is_new_h && (unique_h_count += 1)
        end
        win_unique_resp_p = Float64(unique_p_count)
        win_unique_resp_h = Float64(unique_h_count)
        win_same_port_ratio = same_port_count / win_flow_count
    end

    push!(records, FlowRecord(ts, String(resp_h), String(resp_p)))

    return (win_flow_count, win_unique_resp_p, win_unique_resp_h, win_same_port_ratio)
end

"""
Parameters learned exclusively from IoT-23 training scenarios.
Supports baseline, light, and behavioral feature representations.
"""
struct IoT23Preprocessor{T<:AbstractFloat}
    representation::Symbol
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

function IoT23Preprocessor{T}(
    numeric_names, numeric_means, numeric_scales, numeric_valid_counts,
    categorical_names, categorical_values, categorical_maps,
    feature_names, log_transform, fitted_scenarios,
) where {T<:AbstractFloat}
    n = length(numeric_names)
    rep = n == 7 ? :baseline : (n == 8 ? :light : :behavioral)
    return IoT23Preprocessor{T}(
        rep, numeric_names, numeric_means, numeric_scales, numeric_valid_counts,
        categorical_names, categorical_values, categorical_maps,
        feature_names, log_transform, fitted_scenarios,
    )
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, ::Type{IoT23Preprocessor{T}}) where {T}
    f1 = deserialize(s)
    if f1 isa Symbol
        rep = f1
        num_names = deserialize(s)::Vector{String}
        num_means = deserialize(s)::Vector{Float64}
        num_scales = deserialize(s)::Vector{Float64}
        num_counts = deserialize(s)::Vector{Int}
        cat_names = deserialize(s)::Vector{String}
        cat_vals = deserialize(s)::Vector{Vector{String}}
        cat_maps = deserialize(s)::Vector{Dict{String,Int}}
        feat_names = deserialize(s)::Vector{String}
        log_tf = deserialize(s)::Bool
        fitted = deserialize(s)::Vector{String}
        return IoT23Preprocessor{T}(rep, num_names, num_means, num_scales, num_counts, cat_names, cat_vals, cat_maps, feat_names, log_tf, fitted)
    else
        num_names = f1::Vector{String}
        num_means = deserialize(s)::Vector{Float64}
        num_scales = deserialize(s)::Vector{Float64}
        num_counts = deserialize(s)::Vector{Int}
        cat_names = deserialize(s)::Vector{String}
        cat_vals = deserialize(s)::Vector{Vector{String}}
        cat_maps = deserialize(s)::Vector{Dict{String,Int}}
        feat_names = deserialize(s)::Vector{String}
        log_tf = deserialize(s)::Bool
        fitted = deserialize(s)::Vector{String}
        rep = length(num_names) == 7 ? :baseline : (length(num_names) == 8 ? :light : :behavioral)
        return IoT23Preprocessor{T}(rep, num_names, num_means, num_scales, num_counts, cat_names, cat_vals, cat_maps, feat_names, log_tf, fitted)
    end
end

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

function _bucket_port(port_str::AbstractString)
    cleaned = strip(port_str)
    return cleaned in IOT23_COMMON_PORTS ? String(cleaned) : "__OTHER_PORT__"
end

function _parse_history_flags(history_str::AbstractString)
    has_syn = has_synack = has_ack = has_data = has_fin = has_rst = 0.0
    for char in history_str
        if char == 'S' || char == 's'
            has_syn += 1.0
        elseif char == 'h' || char == 'H'
            has_synack += 1.0
        elseif char == 'A' || char == 'a'
            has_ack += 1.0
        elseif char == 'D' || char == 'd'
            has_data += 1.0
        elseif char == 'F' || char == 'f'
            has_fin += 1.0
        elseif char == 'R' || char == 'r'
            has_rst += 1.0
        end
    end
    return (has_syn, has_synack, has_ack, has_data, has_fin, has_rst)
end

function _iot23_row(line::String)
    empty = SubString(line, 1, 0)
    ts = orig_h = resp_h = resp_p = proto = service = duration = orig_bytes = resp_bytes = conn_state = missed_bytes = history = empty
    orig_pkts = orig_ip_bytes = resp_pkts = resp_ip_bytes = empty
    field = 1
    field_start = 1
    bytes = codeunits(line)

    for position in eachindex(bytes)
        bytes[position] == UInt8(',') || continue
        value = SubString(line, field_start, position - 1)
        field == 1 && (ts = value)
        field == 3 && (orig_h = value)
        field == 5 && (resp_h = value)
        field == 6 && (resp_p = value)
        field == 7 && (proto = value)
        field == 8 && (service = value)
        field == 9 && (duration = value)
        field == 10 && (orig_bytes = value)
        field == 11 && (resp_bytes = value)
        field == 12 && (conn_state = value)
        field == 15 && (missed_bytes = value)
        field == 16 && (history = value)
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
        ts=ts,
        orig_h=orig_h,
        resp_h=resp_h,
        resp_p=resp_p,
        numeric=(duration, orig_bytes, resp_bytes, orig_pkts,
                 resp_pkts, orig_ip_bytes, resp_ip_bytes, missed_bytes),
        history=history,
        categorical=(proto, service, conn_state, _bucket_port(resp_p)),
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

function _extract_raw_numerics(row, representation::Symbol, tracker::OnlineScenarioTracker, row_seq::Float64, log_transform::Bool)
    base_numerics = row.numeric
    if representation == :baseline
        return [_iot23_numeric(base_numerics[i], log_transform) for i in 1:7]
    end

    n_list = [_iot23_numeric(base_numerics[i], log_transform) for i in 1:8]

    if representation == :behavioral
        h_flags = _parse_history_flags(row.history)
        for flag in h_flags
            push!(n_list, (true, log_transform ? log1p(flag) : flag))
        end

        parsed_ts = tryparse(Float64, strip(row.ts))
        ts_val = isnothing(parsed_ts) ? row_seq : parsed_ts
        win_stats = get_temporal_features!(tracker, row.orig_h, row.resp_h, row.resp_p, ts_val)
        for w_val in win_stats
            push!(n_list, (true, log_transform ? log1p(w_val) : w_val))
        end
    end

    return n_list
end

"""
    fit_iot23_preprocessor(train_paths; datatype=Float32, log_transform=true, representation=:baseline)

Learn numerical statistics and categorical vocabularies by streaming only the
given training scenarios. No rows or parameters from a test scenario are used.
"""
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

"""
    fit_all_iot23_preprocessors(train_paths; datatype=Float32, log_transform=true)

Learn numerical statistics and categorical vocabularies for baseline, light,
and behavioral representations in a single streaming pass over the training scenarios.
Returns a Dict{Symbol, IoT23Preprocessor{T}}.
"""
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
    tracker = OnlineScenarioTracker()

    for path in paths
        scenario = basename(path)
        reset!(tracker)
        row_seq = 0.0
        filled = 0
        open(path, "r") do input
            _iot23_header(input, path)
            for line in eachline(input)
                row = _iot23_row(line)
                isnothing(row) && throw(ArgumentError("invalid row structure in $path"))
                row_seq += 1.0
                filled += 1
                _transform_iot23_row!(X, y, filled, row, preprocessor, tracker, row_seq) || throw(ArgumentError(
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
    tracker = OnlineScenarioTracker()

    for path in paths
        scenario = basename(path)
        reset!(tracker)
        row_seq = 0.0
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
                row_seq += 1.0
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
                _transform_iot23_row!(X, y, filled, row, preprocessor, tracker, row_seq) || error(
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
    isfile(path) || throw(ArgumentError("preprocessor file does not exist: $path"))
    try
        open(path, "r") do input
            value = deserialize(input)
            value isa IoT23Preprocessor || throw(ArgumentError("invalid IoT-23 preprocessor file in $path"))
            return value
        end
    catch err
        if err isa EOFError || err isa ArgumentError || err isa MethodError
            throw(ArgumentError("Incompatible preprocessor file format in $path. Run 'julia --project scripts/05_prepare_iot23.jl' to regenerate."))
        else
            rethrow(err)
        end
    end
end
