bert_cp <- RBERT::download_BERT_checkpoint("bert_base_uncased")
vocab_file <- file.path(bert_cp, "vocab.txt")
bert_config_file <- file.path(bert_cp, "bert_config.json")
init_checkpoint <- file.path(bert_cp, "bert_model.ckpt")

trains_trains <- trains_data %>%
  dplyr::filter(mentions_train) %>%
  dplyr::mutate(sequence_index = dplyr::row_number())

layer_output <- RBERT::extract_features(
  RBERT::make_examples_simple(trains_trains$sentence),
  vocab_file = vocab_file,
  bert_config_file = bert_config_file,
  init_checkpoint = init_checkpoint,
  layer_indexes = 0:12,
  features = "output"
)$output

layer_output_labeled <- layer_output %>%
  dplyr::left_join(
    dplyr::select(trains_trains, sequence_index, label),
    by = "sequence_index"
  )

saveRDS(layer_output_labeled, here::here("data", "train_output_labeled.rds"))
