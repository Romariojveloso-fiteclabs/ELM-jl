function _bucket_port(port_str::AbstractString)
    cleaned = strip(port_str)
    return cleaned in IOT23_COMMON_PORTS ? String(cleaned) : "__OTHER_PORT__"
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
