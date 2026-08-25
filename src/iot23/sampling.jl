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
