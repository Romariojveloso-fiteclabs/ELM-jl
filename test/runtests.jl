using ELM
using Test

@testset "ELM classifier" begin
    X = Float32[-2 -1; -1.5 -1.2; -1 -2; 1 1.5; 1.5 1; 2 2]
    y = ["benign", "benign", "benign", "malware", "malware", "malware"]
    model = ELMClassifier(hidden_neurons=20, datatype=Float32, seed=7)
    @test_throws ArgumentError predict(model, X)
    @test fit!(model, X, y) === model
    @test predict(model, X) == y
    @test size(predict_scores(model, X)) == (6, 2)
    @test eltype(model.input_weights) === Float32

    second_model = ELMClassifier(hidden_neurons=20, datatype=Float32, seed=7)
    fit!(second_model, X, y)
    @test second_model.input_weights == model.input_weights
    @test second_model.output_weights == model.output_weights
end

@testset "Preprocessing" begin
    X = [1.0 5.0; 2.0 5.0; 3.0 5.0; 4.0 5.0]
    y = [0, 0, 1, 1]
    normalized, params = normalize_features(X; return_params=true)
    @test normalized[:, 2] == zeros(4)
    @test isapprox(sum(normalized[:, 1]), 0.0; atol=1e-12)
    @test normalize_features(X, params) == normalized
    @test train_test_split(X, y; seed=10) == train_test_split(X, y; seed=10)

    encoded, classes = encode_labels(["a", "b", "a"])
    @test classes == ["a", "b"]
    @test encoded == [1.0 0.0; 0.0 1.0; 1.0 0.0]

    mktemp() do path, io
        write(io, "f1,f2,label\n1.0,2.0,benign\n3.0,4.0,malware\n")
        close(io)
        loaded_X, loaded_y = load_dataset(path; target="label", datatype=Float32)
        @test loaded_X == Float32[1 2; 3 4]
        @test loaded_y == ["benign", "malware"]
    end
end

@testset "Metrics" begin
    truth = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    @test accuracy(truth, predicted) == 0.75
    @test precision(truth, predicted) == 2 / 3
    @test recall(truth, predicted) == 1.0
    @test f1_score(truth, predicted) == 0.8
    multiclass_truth = ["a", "b", "c"]
    @test precision(multiclass_truth, multiclass_truth) == 1.0
    @test recall(multiclass_truth, multiclass_truth) == 1.0
end
