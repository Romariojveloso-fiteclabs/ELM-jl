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
