using DrWatson

@quickactivate "ELM"

using DelimitedFiles
using ELM

if isempty(ARGS)
    println("Usage: julia --project=. scripts/02_security_experiment.jl DATASET.csv [TARGET]")
    println("TARGET may be a one-based column index or header name; default: last column.")
    exit(1)
end

dataset_path = abspath(ARGS[1])
target = if length(ARGS) < 2
    :last
else
    parsed_index = tryparse(Int, ARGS[2])
    isnothing(parsed_index) ? ARGS[2] : parsed_index
end

X, y = load_dataset(dataset_path; target, datatype=Float32)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y; test_size=0.2, seed=42)
Xtrain, normalization = normalize_features(Xtrain; return_params=true)
Xtest = normalize_features(Xtest, normalization)

model = ELMClassifier(
    hidden_neurons=1000,
    activation=abs_activation,
    datatype=Float32,
    lambda=0.01,
    seed=42,
)
fit!(model, Xtrain, ytrain)
ypred = predict(model, Xtest)

metric_names = ["accuracy", "precision", "recall", "f1"]
metric_values = [accuracy(ytest, ypred), precision(ytest, ypred),
                 recall(ytest, ypred), f1_score(ytest, ypred)]
results_directory = plotsdir()
mkpath(results_directory)
results_path = joinpath(results_directory, "security_experiment_metrics.csv")
writedlm(results_path, hcat(metric_names, metric_values), ',')

for (name, value) in zip(metric_names, metric_values)
    println("$(uppercasefirst(name)): $(round(value; digits=4))")
end
println("Results saved to: $results_path")
