using Serialization

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
