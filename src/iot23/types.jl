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

Base.eltype(::IoT23Preprocessor{T}) where {T} = T

struct IoT23SamplingPlan
    scenarios::Vector{String}
    available::Dict{Tuple{String,Int8},Int}
    selected::Dict{Tuple{String,Int8},Int}
    seed::Int
end
