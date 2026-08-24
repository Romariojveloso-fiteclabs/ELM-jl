JULIA ?= julia
PROJECT ?= .

IOT23_VALIDATION_SCENARIO ?= dataset12.csv
IOT23_TEST_SCENARIO ?= dataset23.csv
IOT23_TARGET_PER_CLASS ?= 500000
IOT23_VALIDATION_TARGET_PER_CLASS ?= 100000
IOT23_HIDDEN_NEURONS ?= 512
IOT23_BATCH_SIZE ?= 25000
IOT23_LAMBDA ?= 0.01
IOT23_SEED ?= 42
IOT23_MIN_VALIDATION_RECALL ?=
IOT23_MIN_VALIDATION_SPECIFICITY ?= 0.90

JULIA_CMD = $(JULIA) --project=$(PROJECT)

.DEFAULT_GOAL := help
.PHONY: help makehelp install test validate profile prepare train results retest full

help makehelp:
	@printf '%s\n' \
	  'ELM-jl — comandos disponíveis:' \
	  '  make install   Instala as dependências Julia do projeto.' \
	  '  make test      Executa os testes automatizados.' \
	  '  make validate  Executa a validação sintética rápida da ELM.' \
	  '  make profile   Lê os 23 CSVs e gera o perfil dos dados brutos.' \
	  '  make prepare   Define treino/validação/teste e ajusta o pré-processador.' \
	  '  make train     Treina, calibra o limiar e avalia no teste final.' \
	  '  make results   Exibe métricas, regra de decisão e matriz de confusão.' \
	  '  make retest    Executa testes automatizados, treinamento e resultados.' \
	  '  make full      Executa profile, prepare, train e results (demorado).' \
	  '' \
	  'Exemplos:' \
	  '  make retest' \
	  '  make train IOT23_MIN_VALIDATION_SPECIFICITY=0.95' \
	  '  make train IOT23_TARGET_PER_CLASS=100000 IOT23_HIDDEN_NEURONS=256'

install:
	$(JULIA_CMD) -e 'using Pkg; Pkg.instantiate()'

test:
	$(JULIA_CMD) -e 'using Pkg; Pkg.test()'

validate:
	$(JULIA_CMD) scripts/01_validate_elm.jl

profile:
	$(JULIA_CMD) scripts/04_profile_iot23.jl

prepare:
	IOT23_VALIDATION_SCENARIO="$(IOT23_VALIDATION_SCENARIO)" \
	IOT23_TEST_SCENARIO="$(IOT23_TEST_SCENARIO)" \
	$(JULIA_CMD) scripts/05_prepare_iot23.jl

train:
	IOT23_TARGET_PER_CLASS="$(IOT23_TARGET_PER_CLASS)" \
	IOT23_VALIDATION_TARGET_PER_CLASS="$(IOT23_VALIDATION_TARGET_PER_CLASS)" \
	IOT23_HIDDEN_NEURONS="$(IOT23_HIDDEN_NEURONS)" \
	IOT23_BATCH_SIZE="$(IOT23_BATCH_SIZE)" \
	IOT23_LAMBDA="$(IOT23_LAMBDA)" \
	IOT23_SEED="$(IOT23_SEED)" \
	IOT23_MIN_VALIDATION_RECALL="$(IOT23_MIN_VALIDATION_RECALL)" \
	IOT23_MIN_VALIDATION_SPECIFICITY="$(IOT23_MIN_VALIDATION_SPECIFICITY)" \
	$(JULIA_CMD) scripts/06_train_iot23_batched.jl

results:
	column -s, -t < data/exp_pro/iot23/elm_batched_validation_metrics.csv
	column -s, -t < data/exp_pro/iot23/elm_batched_decision_rule.csv
	column -s, -t < data/exp_pro/iot23/elm_batched_metrics.csv
	column -s, -t < data/exp_pro/iot23/elm_batched_confusion_matrix.csv

retest: test train results

full: profile prepare train results
