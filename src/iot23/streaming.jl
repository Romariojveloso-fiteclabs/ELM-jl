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
