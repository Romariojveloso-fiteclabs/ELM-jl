using DrWatson

@quickactivate "ELM"

using CSV
using DataFrames
using Dates
using ELM
using LinearAlgebra
using Serialization

const TARGET_PER_CLASS = parse(Int, get(ENV, "IOT23_TARGET_PER_CLASS", "500000"))
const VALIDATION_TARGET_PER_CLASS = parse(Int, get(
    ENV, "IOT23_VALIDATION_TARGET_PER_CLASS", "100000",
))
const HIDDEN_NEURONS = parse(Int, get(ENV, "IOT23_HIDDEN_NEURONS", "512"))
const BATCH_SIZE = parse(Int, get(ENV, "IOT23_BATCH_SIZE", "25000"))
const LAMBDA = parse(Float32, get(ENV, "IOT23_LAMBDA", "0.01"))
const SEED = parse(Int, get(ENV, "IOT23_SEED", "42"))
const MINIMUM_VALIDATION_RECALL = let value = get(ENV, "IOT23_MIN_VALIDATION_RECALL", "")
    isempty(value) ? nothing : parse(Float64, value)
end
const MINIMUM_VALIDATION_SPECIFICITY = parse(Float64, get(
    ENV, "IOT23_MIN_VALIDATION_SPECIFICITY", "0.90",
))

function sampling_plan_table(plan::IoT23SamplingPlan)
    rows = NamedTuple[]
    for scenario in plan.scenarios, (label, class_name) in ((Int8(0), "benign"), (Int8(1), "malicious"))
        push!(rows, (
            dataset=scenario,
            class=class_name,
            available=plan.available[(scenario, label)],
            selected=plan.selected[(scenario, label)],
        ))
    end
    return DataFrame(rows)
end

function serialize_atomically(path, value)
    temporary = path * ".tmp"
    open(temporary, "w") do output
        serialize(output, value)
    end
    mv(temporary, path; force=true)
    return path
end

function confusion_counts(y, predicted)
    tn = fp = fn = tp = 0
    for index in eachindex(y)
        truth = y[index]
        prediction = predicted[index]
        if truth == 0 && prediction == 0
            tn += 1
        elseif truth == 0 && prediction == 1
            fp += 1
        elseif truth == 1 && prediction == 0
            fn += 1
        else
            tp += 1
        end
    end
    return (tn=tn, fp=fp, fn=fn, tp=tp)
end

function metric_values(counts)
    total = counts.tn + counts.fp + counts.fn + counts.tp
    total > 0 || error("Matriz de confusão vazia")
    accuracy_value = (counts.tp + counts.tn) / total
    precision_value = iszero(counts.tp + counts.fp) ? 0.0 : counts.tp / (counts.tp + counts.fp)
    recall_value = iszero(counts.tp + counts.fn) ? 0.0 : counts.tp / (counts.tp + counts.fn)
    specificity_value = iszero(counts.tn + counts.fp) ? 0.0 : counts.tn / (counts.tn + counts.fp)
    f1_value = iszero(precision_value + recall_value) ? 0.0 :
               2 * precision_value * recall_value / (precision_value + recall_value)
    return (
        accuracy=accuracy_value,
        precision=precision_value,
        recall=recall_value,
        specificity=specificity_value,
        f1=f1_value,
    )
end

function write_evaluation(output_directory, prefix, counts)
    metrics = metric_values(counts)
    metrics_path = joinpath(output_directory, "$(prefix)_metrics.csv")
    CSV.write(metrics_path, DataFrame(
        metric=["accuracy", "precision", "recall", "specificity", "f1"],
        value=[metrics.accuracy, metrics.precision, metrics.recall, metrics.specificity, metrics.f1],
    ))
    confusion_path = joinpath(output_directory, "$(prefix)_confusion_matrix.csv")
    CSV.write(confusion_path, DataFrame(
        actual=["benign", "malicious"],
        predicted_benign=[counts.tn, counts.fn],
        predicted_malicious=[counts.fp, counts.tp],
    ))
    return merge((metrics_path=metrics_path, confusion_path=confusion_path), metrics)
end

function main()
    raw_directory = datadir("exp_raw", "iot23")
    output_directory = datadir("exp_pro", "iot23")
    split = CSV.read(joinpath(output_directory, "scenario_split.csv"), DataFrame)
    preprocessor_file = get(ENV, "IOT23_PREPROCESSOR_PATH", joinpath(output_directory, "preprocessor.jls"))
    preprocessor = load_iot23_preprocessor(preprocessor_file)

    train = split[split.partition .== "train", :]
    validation = split[split.partition .== "validation", :]
    test = split[split.partition .== "test", :]
    nrow(train) == 21 || error("Esperados 21 cenários de treino; execute scripts/05_prepare_iot23.jl")
    nrow(validation) == 1 || error("Esperado um cenário de validação")
    nrow(test) == 1 || error("Esperado um cenário de teste")
    validation.dataset[1] ∉ preprocessor.fitted_scenarios || error("Vazamento do cenário de validação")
    test.dataset[1] ∉ preprocessor.fitted_scenarios || error("Vazamento do cenário de teste")

    train_paths = joinpath.(raw_directory, train.dataset)
    validation_paths = joinpath.(raw_directory, validation.dataset)
    test_paths = joinpath.(raw_directory, test.dataset)

    plan = balanced_iot23_sampling_plan(
        train.dataset,
        train.benign,
        train.malicious;
        target_per_class=TARGET_PER_CLASS,
        seed=SEED,
    )
    plan_table = sampling_plan_table(plan)
    sampling_path = joinpath(output_directory, "training_sampling_plan.csv")
    CSV.write(sampling_path, plan_table)
    sum(plan_table.selected[plan_table.class .== "benign"]) == TARGET_PER_CLASS || error(
        "Cota benigna incorreta",
    )
    sum(plan_table.selected[plan_table.class .== "malicious"]) == TARGET_PER_CLASS || error(
        "Cota maliciosa incorreta",
    )

    available_validation = min(sum(validation.benign), sum(validation.malicious))
    validation_target = min(VALIDATION_TARGET_PER_CLASS, available_validation)
    validation_target > 0 || error("O cenário de validação deve conter as duas classes")
    validation_plan = balanced_iot23_sampling_plan(
        validation.dataset,
        validation.benign,
        validation.malicious;
        target_per_class=validation_target,
        seed=SEED,
    )
    validation_sampling_path = joinpath(output_directory, "validation_sampling_plan.csv")
    CSV.write(validation_sampling_path, sampling_plan_table(validation_plan))

    model = ELMClassifier(
        hidden_neurons=HIDDEN_NEURONS,
        activation=abs_activation,
        datatype=Float32,
        lambda=LAMBDA,
        seed=SEED,
    )
    accumulator = initialize_batched_fit!(
        model,
        length(preprocessor.feature_names),
        [Int8(0), Int8(1)],
    )

    println("Início: ", now())
    println("BLAS threads: ", BLAS.get_num_threads())
    println("Treino: $(nrow(train)) cenários / amostra=$(2 * TARGET_PER_CLASS) linhas")
    println("Validação: $(only(validation.dataset)) / amostra=$(2 * validation_target) linhas")
    println("Teste final: $(only(test.dataset))")
    println("ELM: $HIDDEN_NEURONS neurônios, lote=$BATCH_SIZE, λ=$LAMBDA, seed=$SEED")
    println("Features: $(length(preprocessor.feature_names)) Float32")
    training_started = time()
    batches = Ref(0)

    sampling_stats = foreach_sampled_iot23_batch(
        train_paths,
        preprocessor,
        plan;
        batch_size=BATCH_SIZE,
    ) do X, y, scenario
        partial_fit!(accumulator, model, X, y)
        batches[] += 1
        if batches[] % 10 == 0
            println(
                "  lotes=$(batches[]) acumulados=$(accumulator.rows) ",
                "cenário=$scenario tempo=$(round(time() - training_started; digits=1))s",
            )
            flush(stdout)
        end
    end
    finalize_batched_fit!(accumulator, model)
    training_seconds = time() - training_started

    sampling_stats.sampled_rows == 2 * TARGET_PER_CLASS || error("Amostra final incorreta")
    accumulator.class_counts == [TARGET_PER_CLASS, TARGET_PER_CLASS] || error(
        "Classes acumuladas incorretamente",
    )
    model_path = joinpath(output_directory, "elm_batched_model.jls")
    serialize_atomically(model_path, model)

    println("\nTreinamento concluído em $(round(training_seconds; digits=1))s")
    println("Calibrando limiar na validação (sem usar o teste final)...")
    validation_started = time()
    validation_scores = Float64[]
    validation_truth = Int8[]
    validation_stats = foreach_sampled_iot23_batch(
        validation_paths,
        preprocessor,
        validation_plan;
        batch_size=BATCH_SIZE,
    ) do X, y, _
        append!(validation_scores, Float64.(binary_decision_scores(model, X; positive_label=Int8(1))))
        append!(validation_truth, y)
    end
    validation_stats.sampled_rows == 2 * validation_target || error("Amostra de validação incorreta")
    calibration = optimal_f1_threshold(
        validation_scores,
        validation_truth;
        positive_label=Int8(1),
        minimum_recall=MINIMUM_VALIDATION_RECALL,
        minimum_specificity=MINIMUM_VALIDATION_SPECIFICITY,
    )
    validation_seconds = time() - validation_started
    validation_counts = (
        tn=calibration.true_negative,
        fp=calibration.false_positive,
        fn=calibration.false_negative,
        tp=calibration.true_positive,
    )
    validation_result = write_evaluation(
        output_directory,
        "elm_batched_validation",
        validation_counts,
    )
    decision_rule_path = joinpath(output_directory, "elm_batched_decision_rule.csv")
    CSV.write(decision_rule_path, DataFrame([(
        positive_label="malicious",
        negative_label="benign",
        threshold=calibration.threshold,
        objective="max_f1_subject_to_constraints",
        minimum_validation_recall=something(MINIMUM_VALIDATION_RECALL, missing),
        minimum_validation_specificity=MINIMUM_VALIDATION_SPECIFICITY,
        validation_scenario=only(validation.dataset),
        validation_sample_per_class=validation_target,
        validation_f1=validation_result.f1,
        validation_precision=validation_result.precision,
        validation_recall=validation_result.recall,
        validation_specificity=validation_result.specificity,
    )]))
    println(
        "Limiar=$(round(calibration.threshold; digits=6)) ",
        "validação: precision=$(round(validation_result.precision; digits=6)) ",
        "recall=$(round(validation_result.recall; digits=6)) ",
        "specificity=$(round(validation_result.specificity; digits=6)) ",
        "F1=$(round(validation_result.f1; digits=6))",
    )

    println("\nAvaliando cenário de teste completo com o limiar calibrado: ", only(test.dataset))
    evaluation_started = time()
    tn = fp = fn = tp = 0
    test_stats = foreach_iot23_batch(test_paths, preprocessor; batch_size=BATCH_SIZE) do X, y, _
        predicted = predict_with_threshold(
            model,
            X,
            calibration.threshold;
            positive_label=Int8(1),
        )
        counts = confusion_counts(y, predicted)
        tn += counts.tn
        fp += counts.fp
        fn += counts.fn
        tp += counts.tp
    end
    evaluation_seconds = time() - evaluation_started
    tn + fp + fn + tp == test_stats.rows || error("Matriz de confusão incompleta")
    test_result = write_evaluation(
        output_directory,
        "elm_batched",
        (tn=tn, fp=fp, fn=fn, tp=tp),
    )

    summary_path = joinpath(output_directory, "elm_batched_training_summary.csv")
    CSV.write(summary_path, DataFrame([(
        train_scenarios=nrow(train),
        validation_scenario=only(validation.dataset),
        test_scenario=only(test.dataset),
        scanned_training_rows=sampling_stats.scanned_rows,
        sampled_training_rows=sampling_stats.sampled_rows,
        sampled_benign=sampling_stats.benign,
        sampled_malicious=sampling_stats.malicious,
        validation_scanned_rows=validation_stats.scanned_rows,
        validation_sampled_rows=validation_stats.sampled_rows,
        validation_target_per_class=validation_target,
        decision_threshold=calibration.threshold,
        minimum_validation_recall=something(MINIMUM_VALIDATION_RECALL, missing),
        minimum_validation_specificity=MINIMUM_VALIDATION_SPECIFICITY,
        hidden_neurons=HIDDEN_NEURONS,
        batch_size=BATCH_SIZE,
        feature_count=length(preprocessor.feature_names),
        lambda=LAMBDA,
        seed=SEED,
        training_seconds=training_seconds,
        validation_seconds=validation_seconds,
        evaluation_seconds=evaluation_seconds,
    )]))

    println("\nTeste concluído em $(round(evaluation_seconds; digits=1))s")
    println("TN=$tn FP=$fp FN=$fn TP=$tp")
    println("Accuracy=$(round(test_result.accuracy; digits=6))")
    println("Precision=$(round(test_result.precision; digits=6))")
    println("Recall=$(round(test_result.recall; digits=6))")
    println("Specificity=$(round(test_result.specificity; digits=6))")
    println("F1=$(round(test_result.f1; digits=6))")
    println("Modelo: $model_path")
    println("Regra de decisão: $decision_rule_path")
    println("Métricas de validação: $(validation_result.metrics_path)")
    println("Métricas de teste: $(test_result.metrics_path)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
