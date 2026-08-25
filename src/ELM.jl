module ELM

using LinearAlgebra
using Random
using Serialization
import Base: precision

include("preprocessing/activations.jl")

include("model/classifier.jl")
include("model/training.jl")
include("model/batch_training.jl")
include("model/prediction.jl")

include("preprocessing/normalization.jl")
include("preprocessing/labels.jl")

include("iot23/types.jl")
include("iot23/schema.jl")
include("iot23/parser.jl")
include("iot23/features.jl")
include("iot23/preprocessor.jl")
include("iot23/streaming.jl")
include("iot23/sampling.jl")
include("iot23/serialization.jl")

include("evaluation/metrics.jl")
include("evaluation/threshold.jl")

export ELMClassifier, fit!, predict, predict_scores, binary_decision_scores,
       predict_with_threshold, optimal_f1_threshold,
       ELMFitAccumulator, initialize_batched_fit!, partial_fit!, finalize_batched_fit!,
       abs_activation, tanh_activation,
       load_dataset, normalize_features, NormalizationParams,
       encode_labels, train_test_split,
       IoT23Preprocessor, fit_iot23_preprocessor, fit_all_iot23_preprocessors,
       foreach_iot23_batch, save_iot23_preprocessor,
       load_iot23_preprocessor, IoT23SamplingPlan,
       balanced_iot23_sampling_plan, foreach_sampled_iot23_batch,
       accuracy, precision, recall, f1_score

end
