bert_cp <- RBERT::download_BERT_checkpoint("bert_base_uncased")
vocab_file <- file.path(bert_cp, "vocab.txt")
bert_config_file <- file.path(bert_cp, "bert_config.json")
init_checkpoint <- file.path(bert_cp, "bert_model.ckpt")

RBERT::extract_features(
  RBERT::make_examples_simple("I love tacos."),
  vocab_file = vocab_file,
  bert_config_file = bert_config_file,
  init_checkpoint = init_checkpoint,
  layer_indexes = 1:12,
  features = "attention"
)$attention %>%
  RBERTviz::visualize_attention() %>%
  htmlwidgets::saveWidget(here::here("tacos_viz.html"))
