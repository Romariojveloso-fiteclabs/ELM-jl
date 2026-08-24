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

    batched_model = ELMClassifier(hidden_neurons=20, datatype=Float32, seed=7)
    accumulator = initialize_batched_fit!(batched_model, size(X, 2), ["benign", "malware"])
    partial_fit!(accumulator, batched_model, X[1:2, :], y[1:2])
    partial_fit!(accumulator, batched_model, X[3:6, :], y[3:6])
    finalize_batched_fit!(accumulator, batched_model)
    @test accumulator.rows == 6
    @test accumulator.class_counts == [3, 3]
    @test predict(batched_model, X) == y

    decision_scores = binary_decision_scores(batched_model, X; positive_label="malware")
    @test length(decision_scores) == length(y)
    @test predict_with_threshold(
        batched_model, X, 0.0; positive_label="malware",
    ) == y

    calibration = optimal_f1_threshold(
        [0.8, 0.4, 0.3, 0.1], [1, 0, 1, 0]; positive_label=1,
    )
    @test isapprox(calibration.threshold, 0.3; atol=1e-12)
    @test isapprox(calibration.f1, 0.8; atol=1e-12)
    @test calibration.true_positive == 2
    @test calibration.false_positive == 1
    @test isapprox(calibration.specificity, 0.5; atol=1e-12)

    specificity_constrained = optimal_f1_threshold(
        [0.8, 0.4, 0.3, 0.1], [1, 0, 1, 0];
        positive_label=1,
        minimum_specificity=0.75,
    )
    @test isapprox(specificity_constrained.threshold, 0.8; atol=1e-12)
    @test specificity_constrained.specificity == 1.0

    constrained = optimal_f1_threshold(
        [0.9, 0.8, 0.7, 0.6], [1, 0, 0, 1];
        positive_label=1,
        minimum_recall=0.75,
    )
    @test isapprox(constrained.threshold, 0.6; atol=1e-12)
    @test constrained.recall == 1.0
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

@testset "IoT-23 preprocessing without leakage" begin
    mktempdir() do directory
        header = join([
            "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
            "proto", "service", "duration", "orig_bytes", "resp_bytes",
            "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
            "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
            "tunnel_parents   label   detailed-label",
        ], ',')
        function row(; proto="tcp", service="dns", duration="1.0", state="SF",
                     label="benign", detail="-")
            return join([
                "1.0", "uid", "10.0.0.1", "123", "10.0.0.2", "80",
                proto, service, duration, "10", "20", state, "-", "-",
                "0", "Dd", "1", "50", "2", "70",
                "-   $label   $detail",
            ], ',')
        end

        train_path = joinpath(directory, "dataset1.csv")
        test_path = joinpath(directory, "dataset23.csv")
        open(train_path, "w") do output
            println(output, header)
            println(output, row())
            println(output, row(duration="-", proto="udp", service="-",
                                state="S0", label="malicious", detail="DDoS"))
        end
        open(test_path, "w") do output
            println(output, header)
            println(output, row(duration="1000000", proto="icmp", service="ssh",
                                state="REJ", label="malicious", detail="Attack"))
        end

        preprocessor = fit_iot23_preprocessor([train_path]; progress=false)
        @test preprocessor.fitted_scenarios == ["dataset1.csv"]
        @test "dataset23.csv" ∉ preprocessor.fitted_scenarios
        @test preprocessor.numeric_valid_counts[1] == 1
        @test preprocessor.numeric_means[1] == log1p(1.0)

        captured_X = Matrix{Float32}[]
        captured_y = Vector{Int8}[]
        stats = foreach_iot23_batch([test_path], preprocessor; batch_size=1) do X, y, _
            push!(captured_X, copy(X))
            push!(captured_y, copy(y))
        end
        @test stats == (rows=1, benign=0, malicious=1)
        @test captured_y == [Int8[1]]
        @test all(isfinite, only(captured_X))
        @test only(captured_X)[1, findfirst(==("proto=__UNKNOWN__"), preprocessor.feature_names)] == 1
        @test only(captured_X)[1, findfirst(==("service=__UNKNOWN__"), preprocessor.feature_names)] == 1
        @test only(captured_X)[1, findfirst(==("conn_state=__UNKNOWN__"), preprocessor.feature_names)] == 1

        saved_path = joinpath(directory, "preprocessor.jls")
        save_iot23_preprocessor(saved_path, preprocessor)
        restored = load_iot23_preprocessor(saved_path)
        @test restored.feature_names == preprocessor.feature_names
        @test eltype(restored) === Float32

        plan = balanced_iot23_sampling_plan(
            ["dataset1.csv"], [1], [1]; target_per_class=1, seed=42,
        )
        sampled_labels = Int8[]
        sampled = foreach_sampled_iot23_batch(
            (X, y, _) -> append!(sampled_labels, y),
            [train_path],
            preprocessor,
            plan;
            batch_size=1,
        )
        @test sampled.scanned_rows == 2
        @test sampled.sampled_rows == 2
        @test sampled.benign == sampled.malicious == 1
        @test sort(sampled_labels) == Int8[0, 1]
    end
end
