# ELM-jl

Implementação em Julia de uma **Extreme Learning Machine (ELM)** para a
classificação binária de tráfego da base IoT-23. O pipeline foi preparado para
ler os CSVs em lotes, sem carregar dezenas de gigabytes simultaneamente na RAM.

Por padrão, o experimento usa 21 cenários para treino, reserva `dataset12.csv`
para validação e mantém integralmente `dataset23.csv` para teste final. Assim,
os parâmetros de pré-processamento e o limiar de decisão são definidos sem usar
o teste final.

## Requisitos

- Julia 1.10 ou superior (testado também com Julia 1.12).
- Os 23 arquivos CSV da IoT-23 extraídos localmente.
- Espaço em disco para os dados brutos: a extração completa ocupa cerca de
  45 GB. O pipeline não cria uma cópia pré-processada completa desses CSVs.

Confira a instalação do Julia:

```bash
julia --version
```

## Instalação do ambiente

Clone o repositório (ou entre no diretório já existente) e instale as
dependências declaradas em `Project.toml`:

```bash
cd /caminho/para/ELM-jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

Para executar os testes automatizados:

```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

Também há um `Makefile` para encurtar os comandos. Veja as opções disponíveis:

```bash
make help
```

Uma validação rápida da ELM com dados sintéticos pode ser feita com:

```bash
julia --project=. scripts/01_validate_elm.jl
```

## Dados da IoT-23

Os CSVs devem estar exatamente neste diretório, com os nomes originais:

```text
data/exp_raw/iot23/
├── dataset1.csv
├── dataset2.csv
├── ...
└── dataset23.csv
```

Antes de iniciar o pipeline, confira se há os 23 arquivos:

```bash
find data/exp_raw/iot23 -maxdepth 1 -type f -name 'dataset*.csv' | wc -l
```

O resultado esperado é `23`.

## Fluxo completo de treinamento

Execute os comandos a seguir a partir da raiz do projeto, na ordem indicada.
Os artefatos intermediários e resultados são gravados em
`data/exp_pro/iot23/`.

### 1. Perfil dos dados brutos

O perfil verifica os cenários, colunas, rótulos e suas distribuições. É
necessário apenas antes da primeira preparação, ou se os CSVs forem alterados.

```bash
julia --project=. scripts/04_profile_iot23.jl
```

Arquivos gerados:

- `profile.csv`: quantidade de linhas e distribuição benigno/malicioso por cenário.
- `detailed_labels.csv`: rótulos detalhados encontrados nos dados.

Para apenas inspecionar um CSV de forma exploratória, há também:

```bash
julia --project=. scripts/03_inspect_iot23.jl
```

### 2. Definir treino/teste e ajustar o pré-processamento

```bash
julia --project=. scripts/05_prepare_iot23.jl
```

Este passo reserva `dataset12.csv` para validação, `dataset23.csv` para teste e
usa os outros 21 cenários no treino. Ele ajusta `log1p`, normalização e
vocabulários das variáveis categóricas somente no treino. Categorias novas na
validação ou no teste são codificadas como `__UNKNOWN__`.

Para escolher outro cenário de validação, execute a preparação com a variável
de ambiente abaixo. O cenário de teste também é configurável, mas não deve ser
alterado durante uma comparação de experimentos.

```bash
IOT23_VALIDATION_SCENARIO=dataset12.csv \
IOT23_TEST_SCENARIO=dataset23.csv \
julia --project=. scripts/05_prepare_iot23.jl
```

Arquivos gerados:

- `scenario_split.csv`: divisão dos cenários em treino e teste.
- `preprocessor.jls`: pré-processador ajustado no treino.
- `preprocessor_features.csv`: lista e descrição das features produzidas.
- `preprocessing_validation.csv`: verificação do isolamento entre treino, validação e teste.

### 3. Treinar a ELM em lotes e avaliar

```bash
julia --project=. scripts/06_train_iot23_batched.jl
```

Por padrão, o treinamento seleciona uniformemente 500.000 linhas benignas e
500.000 maliciosas dos 21 cenários de treino, totalizando 1.000.000 de linhas.
Depois, seleciona 100.000 linhas de cada classe do cenário de validação. Os
escores da ELM nessa amostra definem automaticamente o limiar que maximiza F1,
desde que mantenha pelo menos 90% de especificidade na validação. Isso impede a
regra degenerada que classifica todo o tráfego como malicioso. O `dataset23.csv`
continua sem participar dessa escolha e mantém sua distribuição original na
avaliação final.

O leitor percorre os dados em lotes de 25.000 linhas e a ELM acumula as
estatísticas necessárias ao ajuste sem reter todos os lotes na memória. Para
garantir uma amostragem uniforme entre todos os cenários, a execução ainda lê
todas as linhas brutas de treino; reduzir o tamanho da amostra não elimina essa
varredura de disco.

Arquivos gerados ou atualizados:

- `training_sampling_plan.csv`: quantidade disponível e selecionada por cenário e classe.
- `validation_sampling_plan.csv`: amostra balanceada usada apenas para calibrar o limiar.
- `elm_batched_model.jls`: modelo ELM treinado.
- `elm_batched_decision_rule.csv`: limiar calibrado e métricas de validação.
- `elm_batched_validation_metrics.csv` e `elm_batched_validation_confusion_matrix.csv`:
  resultados da validação.
- `elm_batched_metrics.csv`: accuracy, precision, recall, especificidade e F1.
- `elm_batched_confusion_matrix.csv`: matriz de confusão do teste.
- `elm_batched_training_summary.csv`: parâmetros, tempo e totais do experimento.

Para visualizar os resultados no terminal:

```bash
column -s, -t < data/exp_pro/iot23/elm_batched_metrics.csv
column -s, -t < data/exp_pro/iot23/elm_batched_confusion_matrix.csv
```

## Parâmetros de treinamento

O script de treinamento aceita variáveis de ambiente. Os valores padrão são:

| Variável | Padrão | Significado |
| --- | ---: | --- |
| `IOT23_TARGET_PER_CLASS` | `500000` | linhas benignas e maliciosas selecionadas no treino |
| `IOT23_VALIDATION_TARGET_PER_CLASS` | `100000` | máximo de linhas por classe usadas na calibração |
| `IOT23_HIDDEN_NEURONS` | `512` | neurônios da camada oculta da ELM |
| `IOT23_BATCH_SIZE` | `25000` | linhas processadas por lote |
| `IOT23_LAMBDA` | `0.01` | regularização ridge da saída da ELM |
| `IOT23_SEED` | `42` | semente para pesos e amostragem reproduzíveis |
| `IOT23_MIN_VALIDATION_RECALL` | vazio | recall mínimo opcional para o limiar; entre `0` e `1` |
| `IOT23_MIN_VALIDATION_SPECIFICITY` | `0.90` | especificidade mínima exigida para o limiar; entre `0` e `1` |

Exemplo com uma amostra menor e uma ELM menor:

```bash
IOT23_TARGET_PER_CLASS=100000 \
IOT23_VALIDATION_TARGET_PER_CLASS=50000 \
IOT23_HIDDEN_NEURONS=256 \
IOT23_BATCH_SIZE=25000 \
IOT23_LAMBDA=0.01 \
IOT23_SEED=42 \
julia --project=. scripts/06_train_iot23_batched.jl
```

Para priorizar um recall mínimo de 80% na validação, mantendo dentre os limiares
elegíveis aquele com maior F1:

```bash
IOT23_MIN_VALIDATION_RECALL=0.80 \
julia --project=. scripts/06_train_iot23_batched.jl
```

Para tornar a regra ainda mais conservadora contra falsos positivos, por
exemplo exigindo especificidade mínima de 95% na validação:

```bash
IOT23_MIN_VALIDATION_SPECIFICITY=0.95 \
julia --project=. scripts/06_train_iot23_batched.jl
```

Mesmo em uma execução reduzida, mantenha `scripts/05_prepare_iot23.jl` já
executado. Para repetir experimentos sem perder resultados anteriores, copie ou
renomeie os arquivos `elm_batched_*` antes de iniciar um novo treinamento, pois
eles são sobrescritos.

## Interpretação dos resultados

Use `elm_batched_validation_metrics.csv` para verificar a qualidade do limiar e
`elm_batched_metrics.csv` apenas como resultado final imparcial do
`dataset23.csv`. Ao comparar experimentos, priorize recall e F1 juntamente com
a matriz de confusão e especificidade, e não apenas accuracy. Um aumento de
recall normalmente eleva os falsos positivos e pode reduzir a precisão; o
limiar existe justamente para controlar esse compromisso.

## Sequência com `make`

Após a primeira instalação, a sequência completa é:

```bash
make profile
make prepare
make train
make results
```

`make full` executa essa sequência inteira. Para repetir somente o experimento
após uma alteração no treinamento — como esta calibração por especificidade —,
o pré-processador já existente pode ser reutilizado:

```bash
make test
make train
make results
```

O atalho `make retest` executa esses três últimos comandos em sequência. Os
parâmetros também podem ser alterados diretamente no `make`:

```bash
make train IOT23_MIN_VALIDATION_SPECIFICITY=0.95
```
