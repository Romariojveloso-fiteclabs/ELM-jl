using DrWatson

@quickactivate "ELM"

using CSV
using DataFrames
using Dates

const EXPECTED_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    "tunnel_parents   label   detailed-label",
]

mutable struct NumericColumnProfile
    dash::Int
    question::Int
    empty::Int
    missing::Int
    other_invalid::Int
end

NumericColumnProfile() = NumericColumnProfile(0, 0, 0, 0, 0)

function observe_numeric!(profile::NumericColumnProfile, text::AbstractString)
    text = strip(text)
    if isempty(text)
        profile.empty += 1
    elseif text == "-"
        profile.dash += 1
    elseif text == "?"
        profile.question += 1
    elseif tryparse(Float64, text) === nothing
        profile.other_invalid += 1
    end
    return
end

mutable struct ScenarioAccumulator
    rows::Int
    benign::Int
    malicious::Int
    unknown_label::Int
    label_parse_errors::Int
    proto_values::Set{String}
    service_values::Set{String}
    conn_state_values::Set{String}
    detailed_labels::Dict{String,Int}
    duration::NumericColumnProfile
    orig_bytes::NumericColumnProfile
    resp_bytes::NumericColumnProfile
end

ScenarioAccumulator() = ScenarioAccumulator(
    0, 0, 0, 0, 0,
    Set{String}(), Set{String}(), Set{String}(), Dict{String,Int}(),
    NumericColumnProfile(), NumericColumnProfile(), NumericColumnProfile(),
)

function observe_category!(values::Set{String}, text::AbstractString)
    token = strip(text)
    token in values || push!(values, String(token))
    return
end

function increment_label!(counts::Dict{String,Int}, label::AbstractString)
    canonical = getkey(counts, label, nothing)
    if isnothing(canonical)
        counts[String(label)] = 1
    else
        counts[canonical] += 1
    end
    return
end

function observe_label!(accumulator::ScenarioAccumulator, text::AbstractString)
    parts = split(strip(text); limit=3)
    if length(parts) != 3
        accumulator.label_parse_errors += 1
        return
    end

    label = parts[2]
    detailed_label = strip(parts[3])
    if label == "benign" || label == "Benign"
        accumulator.benign += 1
    elseif label == "malicious" || label == "Malicious"
        accumulator.malicious += 1
    else
        accumulator.unknown_label += 1
    end
    increment_label!(accumulator.detailed_labels, detailed_label)
    return
end

function observe_field!(
    accumulator::ScenarioAccumulator,
    line::String,
    field::Int,
    first_byte::Int,
    last_byte::Int,
)
    last_byte < first_byte && return
    value = @view line[first_byte:last_byte]
    if field == 7
        observe_category!(accumulator.proto_values, value)
    elseif field == 8
        observe_category!(accumulator.service_values, value)
    elseif field == 9
        observe_numeric!(accumulator.duration, value)
    elseif field == 10
        observe_numeric!(accumulator.orig_bytes, value)
    elseif field == 11
        observe_numeric!(accumulator.resp_bytes, value)
    elseif field == 12
        observe_category!(accumulator.conn_state_values, value)
    end
    return
end

function observe_line!(accumulator::ScenarioAccumulator, line::String)
    accumulator.rows += 1
    bytes = codeunits(line)
    field = 1
    field_start = 1

    for position in eachindex(bytes)
        if bytes[position] == UInt8(',')
            observe_field!(accumulator, line, field, field_start, position - 1)
            field += 1
            field_start = position + 1
        end
    end

    line_end = lastindex(bytes)
    line_end >= field_start && bytes[line_end] == UInt8('\r') && (line_end -= 1)
    if field == 21
        observe_label!(accumulator, @view line[field_start:line_end])
    else
        accumulator.label_parse_errors += 1
    end
    return
end

function inferred_schema(path)
    sample = CSV.File(path; limit=10_000)
    column_names = string.(propertynames(sample))
    column_types = [string(eltype(getproperty(sample, name))) for name in propertynames(sample)]
    return column_names, join(
        ("$(name)::$(type)" for (name, type) in zip(column_names, column_types)),
        ";",
    )
end

function profile_dataset(path)
    started_at = time()
    column_names, types = inferred_schema(path)

    accumulator = ScenarioAccumulator()
    open(path, "r") do input
        eof(input) && error("Arquivo vazio: $path")
        readline(input)
        for line in eachline(input)
            observe_line!(accumulator, line)
        end
    end

    summary = (
        dataset=basename(path),
        size_bytes=filesize(path),
        rows=accumulator.rows,
        columns=length(column_names),
        schema_matches=column_names == EXPECTED_COLUMNS,
        inferred_types=types,
        benign=accumulator.benign,
        malicious=accumulator.malicious,
        unknown_label=accumulator.unknown_label,
        label_parse_errors=accumulator.label_parse_errors,
        detailed_label_classes=length(accumulator.detailed_labels),
        proto_values=join(sort!(collect(accumulator.proto_values)), "|"),
        service_values=join(sort!(collect(accumulator.service_values)), "|"),
        conn_state_values=join(sort!(collect(accumulator.conn_state_values)), "|"),
        duration_dash=accumulator.duration.dash,
        duration_question=accumulator.duration.question,
        duration_empty=accumulator.duration.empty,
        duration_missing=accumulator.duration.missing,
        duration_other_invalid=accumulator.duration.other_invalid,
        orig_bytes_dash=accumulator.orig_bytes.dash,
        orig_bytes_question=accumulator.orig_bytes.question,
        orig_bytes_empty=accumulator.orig_bytes.empty,
        orig_bytes_missing=accumulator.orig_bytes.missing,
        orig_bytes_other_invalid=accumulator.orig_bytes.other_invalid,
        resp_bytes_dash=accumulator.resp_bytes.dash,
        resp_bytes_question=accumulator.resp_bytes.question,
        resp_bytes_empty=accumulator.resp_bytes.empty,
        resp_bytes_missing=accumulator.resp_bytes.missing,
        resp_bytes_other_invalid=accumulator.resp_bytes.other_invalid,
        elapsed_seconds=round(time() - started_at; digits=3),
    )

    label_rows = [
        (dataset=basename(path), detailed_label=label, count=count)
        for (label, count) in sort!(collect(accumulator.detailed_labels); by=first)
    ]
    return summary, label_rows
end

function dataset_number(path)
    matched = match(r"dataset(\d+)\.csv$", basename(path))
    return isnothing(matched) ? typemax(Int) : parse(Int, only(matched.captures))
end

function main()
    input_directory = datadir("exp_raw", "iot23")
    output_directory = datadir("exp_pro", "iot23")
    isdir(input_directory) || error("Diretório IoT-23 não encontrado: $input_directory")
    mkpath(output_directory)

    files = sort(
        filter(path -> occursin(r"dataset\d+\.csv$", basename(path)),
               readdir(input_directory; join=true));
        by=dataset_number,
    )
    length(files) == 23 || error("Esperados 23 CSVs, encontrados $(length(files))")

    profile_path = joinpath(output_directory, "profile.csv")
    labels_path = joinpath(output_directory, "detailed_labels.csv")
    profile_temporary = profile_path * ".tmp"
    labels_temporary = labels_path * ".tmp"
    rm(profile_temporary; force=true)
    rm(labels_temporary; force=true)

    println("Início: ", now())
    println("Cenários encontrados: ", length(files))
    println("Leitura em streaming; um cenário por vez.\n")

    for (index, path) in enumerate(files)
        size_gib = round(filesize(path) / 1024^3; digits=2)
        println("[$index/$(length(files))] $(basename(path)) ($(size_gib) GiB)")
        flush(stdout)

        summary, label_rows = profile_dataset(path)
        CSV.write(
            profile_temporary,
            DataFrame([summary]);
            append=isfile(profile_temporary),
            writeheader=!isfile(profile_temporary),
        )
        if !isempty(label_rows)
            CSV.write(
                labels_temporary,
                DataFrame(label_rows);
                append=isfile(labels_temporary),
                writeheader=!isfile(labels_temporary),
            )
        end
        println(
            "  linhas=$(summary.rows) benign=$(summary.benign) ",
            "malicious=$(summary.malicious) erros=$(summary.label_parse_errors) ",
            "tempo=$(summary.elapsed_seconds)s",
        )
        flush(stdout)
    end

    mv(profile_temporary, profile_path; force=true)
    mv(labels_temporary, labels_path; force=true)
    println("\nFim: ", now())
    println("Perfil: ", profile_path)
    println("Rótulos detalhados: ", labels_path)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
