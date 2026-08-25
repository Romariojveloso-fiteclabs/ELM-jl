mutable struct ELMClassifier{T<:AbstractFloat,F}
    hidden_neurons::Int
    activation::F
    lambda::T
    seed::Int
    input_weights::Matrix{T}
    biases::Vector{T}
    output_weights::Matrix{T}
    classes::Any
    input_size::Int
end

function ELMClassifier(;
    hidden_neurons::Integer=100,
    activation::F=abs_activation,
    datatype::Type{T}=Float64,
    lambda::Real=0.01,
    seed::Integer=42,
) where {T<:AbstractFloat,F}
    hidden_neurons > 0 || throw(ArgumentError("hidden_neurons must be positive"))
    lambda >= 0 || throw(ArgumentError("lambda must be non-negative"))
    return ELMClassifier{T,F}(
        Int(hidden_neurons), activation, T(lambda), Int(seed),
        Matrix{T}(undef, 0, 0), Vector{T}(undef, 0),
        Matrix{T}(undef, 0, 0), nothing, 0,
    )
end
