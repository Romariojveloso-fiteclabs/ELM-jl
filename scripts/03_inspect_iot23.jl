using DrWatson

@quickactivate "ELM"

using CSV
using DataFrames

iot23_dir = datadir("exp_raw", "iot23")
isdir(iot23_dir) || error("Diretório IoT-23 não encontrado: $iot23_dir")

files = sort(filter(
    file -> endswith(lowercase(file), ".csv"),
    readdir(iot23_dir; join=true),
))

println("Arquivos CSV encontrados: ", length(files))
isempty(files) && error("Nenhum arquivo CSV encontrado em $iot23_dir")

file = first(files)
println("\nArquivo analisado:")
println(file)

df = CSV.read(file, DataFrame)

println("\nDimensões:")
println("Linhas: ", nrow(df))
println("Colunas: ", ncol(df))

println("\nColunas:")
println(names(df))

println("\nTipos e valores ausentes:")
for column in names(df)
    println(
        column,
        " | type = ", eltype(df[!, column]),
        " | missing = ", count(ismissing, df[!, column]),
    )
end

println("\nPrimeiras linhas:")
println(first(df, 5))

println("\nDescrição:")
println(describe(df))
