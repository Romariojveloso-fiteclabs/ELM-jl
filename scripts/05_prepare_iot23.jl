using DrWatson

@quickactivate "ELM"

using CSV
using DataFrames
using ELM
using Dates

const VALIDATION_SCENARIO = get(ENV, "IOT23_VALIDATION_SCENARIO", "dataset12.csv")
const TEST_SCENARIO = get(ENV, "IOT23_TEST_SCENARIO", "dataset23.csv")
const REP_SETTING = get(ENV, "IOT23_REPRESENTATION", "all")

function dataset_number(name)
    matched = match(r"dataset(\d+)\.csv$", name)
    return isnothing(matched) ? typemax(Int) : parse(Int, only(matched.captures))
end

function feature_table(preprocessor)
    rows = NamedTuple[]
    numeric_count = length(preprocessor.numeric_names)
    for index in 1:numeric_count
        push!(rows, (
            feature_index=index,
            feature_name=preprocessor.feature_names[index],
            kind="numeric",
            source=preprocessor.numeric_names[index],
            training_mean=preprocessor.numeric_means[index],
            training_scale=preprocessor.numeric_scales[index],
            training_valid_count=preprocessor.numeric_valid_counts[index],
        ))
        push!(rows, (
            feature_index=numeric_count + index,
            feature_name=preprocessor.feature_names[numeric_count + index],
            kind="missing_indicator",
            source=preprocessor.numeric_names[index],
            training_mean=missing,
            training_scale=missing,
            training_valid_count=preprocessor.numeric_valid_counts[index],
        ))
    end

    output_index = 2 * numeric_count + 1
    for (source, values) in zip(
        preprocessor.categorical_names,
        preprocessor.categorical_values,
    )
        for value in vcat(values, ["__UNKNOWN__"])
            push!(rows, (
                feature_index=output_index,
                feature_name=preprocessor.feature_names[output_index],
                kind="one_hot",
                source=source,
                training_mean=missing,
                training_scale=missing,
                training_valid_count=missing,
            ))
            output_index += 1
        end
    end
    return sort!(DataFrame(rows), :feature_index)
end

function validate_batch(X, y, scenario, preprocessor)
    all(isfinite, X) || error("Valores não finitos após transformar $scenario")
    all(label -> label in (0, 1), y) || error("Rótulo não binário em $scenario")

    numeric_count = length(preprocessor.numeric_names)
    indicators = @view X[:, (numeric_count + 1):(2 * numeric_count)]
    all(value -> value == 0 || value == 1, indicators) || error(
        "Indicador de ausência inválido em $scenario",
    )

    offset = 2 * numeric_count + 1
    for values in preprocessor.categorical_values
        width = length(values) + 1
        block = @view X[:, offset:(offset + width - 1)]
        all(sum(block; dims=2) .== 1) || error("One-hot inválido em $scenario")
        offset += width
    end
    return
end

function save_and_validate_preprocessor(preprocessor, train_paths, test_paths, validation_names, test_names, split, raw_directory, output_directory)
    representation_symbol = preprocessor.representation
    rep_str = string(representation_symbol)
    println("\n=== Salvando e Validando Representação: $rep_str ===")

    validation_scenario = only(validation_names)
    test_scenario = only(test_names)
    validation_scenario in preprocessor.fitted_scenarios && error("Vazamento do cenário de validação")
    test_scenario in preprocessor.fitted_scenarios && error("Vazamento do cenário de teste")

    model_path = joinpath(output_directory, "preprocessor_$(rep_str).jls")
    temporary_model_path = model_path * ".tmp"
    save_iot23_preprocessor(temporary_model_path, preprocessor)
    restored = load_iot23_preprocessor(temporary_model_path)
    restored.feature_names == preprocessor.feature_names || error(
        "Falha ao restaurar o pré-processador",
    )

    println("Validando transformação completa do cenário de teste ($test_scenario) em lotes...")
    test_stats = foreach_iot23_batch(
        (X, y, scenario) -> validate_batch(X, y, scenario, restored),
        test_paths,
        restored;
        batch_size=100_000,
    )
    expected_test = split[split.partition .== "test", :]
    test_stats.rows == sum(expected_test.rows) || error("Total de teste divergente")
    test_stats.benign == sum(expected_test.benign) || error("Benignos de teste divergentes")
    test_stats.malicious == sum(expected_test.malicious) || error("Maliciosos de teste divergentes")

    mv(temporary_model_path, model_path; force=true)

    # Copiar também para o preprocessor.jls padrao se for baseline ou behavioral
    if representation_symbol == :baseline || representation_symbol == :behavioral
        cp(model_path, joinpath(output_directory, "preprocessor.jls"); force=true)
    end

    features_path = joinpath(output_directory, "preprocessor_features_$(rep_str).csv")
    CSV.write(features_path, feature_table(preprocessor))
    validation_path = joinpath(output_directory, "preprocessing_validation_$(rep_str).csv")
    CSV.write(validation_path, DataFrame([(
        validation_scenario=only(validation_names),
        test_scenario=only(test_names),
        rows=test_stats.rows,
        benign=test_stats.benign,
        malicious=test_stats.malicious,
        feature_count=length(preprocessor.feature_names),
        datatype=string(eltype(preprocessor)),
        log_transform=preprocessor.log_transform,
        fitted_scenarios=length(preprocessor.fitted_scenarios),
        leakage_check="passed_validation_and_test_excluded",
    )]))

    println("Features produzidas ($(rep_str)): ", length(preprocessor.feature_names))
    println("Pré-processador: ", model_path)
    println("Descrição das features: ", features_path)
    println("Validação: ", validation_path)
end

function main()
    raw_directory = datadir("exp_raw", "iot23")
    output_directory = datadir("exp_pro", "iot23")
    profile_path = joinpath(output_directory, "profile.csv")
    isfile(profile_path) || error("Execute scripts/04_profile_iot23.jl primeiro")

    profile = CSV.read(profile_path, DataFrame)
    nrow(profile) == 23 || error("O perfil deve conter 23 cenários")
    validation_scenario = VALIDATION_SCENARIO
    test_scenario = TEST_SCENARIO
    validation_scenario != test_scenario || error("Validação e teste devem ser cenários distintos")
    validation_scenario in profile.dataset || error("Cenário de validação ausente: $validation_scenario")
    test_scenario in profile.dataset || error("Cenário de teste ausente: $test_scenario")

    sort!(profile, :dataset, by=dataset_number)
    profile.partition = [
        dataset == test_scenario ? "test" :
        dataset == validation_scenario ? "validation" : "train"
        for dataset in profile.dataset
    ]
    split = select(
        profile,
        :dataset,
        :partition,
        :rows,
        :benign,
        :malicious,
    )
    split_path = joinpath(output_directory, "scenario_split.csv")
    CSV.write(split_path, split)

    train_names = split.dataset[split.partition .== "train"]
    validation_names = split.dataset[split.partition .== "validation"]
    test_names = split.dataset[split.partition .== "test"]
    train_paths = joinpath.(raw_directory, train_names)
    train_rows = sum(split.rows[split.partition .== "train"])
    validation_rows = sum(split.rows[split.partition .== "validation"])
    test_paths = joinpath.(raw_directory, test_names)

    println("Início: ", now())
    println("Treino: $(length(train_paths)) cenários / $train_rows linhas")
    println("Validação: $(only(validation_names)) / $validation_rows linhas")
    println("Teste:     $(only(test_names))")

    if REP_SETTING == "all"
        println("\nAjustando todas as representações (:baseline, :light, :behavioral) em uma única passagem pelos dados de treino...")
        preprocessors = fit_all_iot23_preprocessors(
            train_paths;
            datatype=Float32,
            log_transform=true,
            progress=true,
        )
        for rep in (:baseline, :light, :behavioral)
            save_and_validate_preprocessor(preprocessors[rep], train_paths, test_paths, validation_names, test_names, split, raw_directory, output_directory)
        end
    else
        rep = Symbol(REP_SETTING)
        println("\nAjustando representação especificada: $rep...")
        preprocessor = fit_iot23_preprocessor(
            train_paths;
            datatype=Float32,
            log_transform=true,
            representation=rep,
            progress=true,
        )
        save_and_validate_preprocessor(preprocessor, train_paths, test_paths, validation_names, test_names, split, raw_directory, output_directory)
    end

    println("\nFim: ", now())
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
