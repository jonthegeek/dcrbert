bert_cp <- RBERT::download_BERT_checkpoint("bert_base_uncased")
vocab_file <- file.path(bert_cp, "vocab.txt")
bert_config_file <- file.path(bert_cp, "bert_config.json")
init_checkpoint <- file.path(bert_cp, "bert_model.ckpt")

sentences <- c(
  "The chicken didn't cross the road because it was too tired.",
  "The chicken didn't cross the road because it was too wide.",
  "The dog fetched the ball. It was excited.",
  "The dog fetched the ball. It was blue."
)

attention <- RBERT::extract_features(
  RBERT::make_examples_simple(sentences),
  vocab_file = vocab_file,
  bert_config_file = bert_config_file,
  init_checkpoint = init_checkpoint,
  layer_indexes = 1:12,
  features = "attention"
)$attention

RBERTviz::visualize_attention(attention = attention, sequence_index = 1)
RBERTviz::visualize_attention(attention = attention, sequence_index = 2)
RBERTviz::visualize_attention(attention = attention, sequence_index = 3)
RBERTviz::visualize_attention(attention = attention, sequence_index = 4)
