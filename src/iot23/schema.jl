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

function _iot23_header(input, path)
    eof(input) && throw(ArgumentError("empty IoT-23 file: $path"))
    header = readline(input)
    endswith(header, '\r') && (header = chop(header))
    header == IOT23_EXPECTED_HEADER || throw(ArgumentError(
        "unexpected IoT-23 schema in $path",
    ))
    return
end
