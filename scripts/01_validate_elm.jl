using DrWatson

@quickactivate "ELM"

using ELM
using Random

rng = MersenneTwister(42)
samples_per_class = 100
class_zero = randn(rng, Float32, samples_per_class, 2) .* 0.35f0 .- 1.0f0
class_one = randn(rng, Float32, samples_per_class, 2) .* 0.35f0 .+ 1.0f0
X = vcat(class_zero, class_one)
y = vcat(fill(0, samples_per_class), fill(1, samples_per_class))

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y; test_size=0.25, seed=42)
Xtrain, normalization = normalize_features(Xtrain; return_params=true)
Xtest = normalize_features(Xtest, normalization)

model = ELMClassifier(
    hidden_neurons=100,
    activation=abs_activation,
    datatype=Float32,
    lambda=0.01,
    seed=42,
)
fit!(model, Xtrain, ytrain)
ypred = predict(model, Xtest)

println("Validation samples: $(length(ytest))")
println("Accuracy:  $(round(accuracy(ytest, ypred); digits=4))")
println("Precision: $(round(precision(ytest, ypred); digits=4))")
println("Recall:    $(round(recall(ytest, ypred); digits=4))")
println("F1:        $(round(f1_score(ytest, ypred); digits=4))")
