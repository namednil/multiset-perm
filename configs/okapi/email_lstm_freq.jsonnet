{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "em_epochs": 150,
        "end_symbol": "</s>",
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/email/train.tsv",
            "non_copyable": [
                "GET",
                "FILTER",
                "SEARCH",
                "eq",
                "gt",
                "lt",
                "True",
                "False",
                "ORDERBY",
                "COUNT",
                "asc",
                "desc",
                "TOP"
            ],
            "source_tokenizer": {
                "type": "spacy"
            },
            "target_tokenizer": {
                "type": "spacy"
            }
        },
        "source_add_end_token": true,
        "source_add_start_token": false,
        "source_tokenizer": {
            "type": "spacy"
        },
        "start_symbol": "<s>",
        "target_tokenizer": {
            "type": "spacy"
        }
    },
    "model": {
        "type": "multiset_pred",
        "alignment_loss_weight": 0.481479,
        "alignment_threshold": 0.815687,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 100,
            "input_size": 200,
            "layer_dropout_probability": 0.143833,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.143833
        },
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/email/train.tsv",
            "non_copyable": [
                "GET",
                "FILTER",
                "SEARCH",
                "eq",
                "gt",
                "lt",
                "True",
                "False",
                "ORDERBY",
                "COUNT",
                "asc",
                "desc",
                "TOP"
            ],
            "source_tokenizer": {
                "type": "spacy"
            },
            "target_tokenizer": {
                "type": "spacy"
            }
        },
        "lower_case_eval": true,
        "max_n": 4,
        "mlp": {
            "activations": "gelu",
            "dropout": 0.512478,
            "hidden_dims": 1024,
            "input_dim": 200,
            "num_layers": 1
        },
        "pretrain_epochs": 8,
        "rho": 1,
        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 200,
                    "pretrained_file": "~/downloads/glove.6B.200d.txt",
                    "trainable": true
                }
            }
        }
    },
    "train_data_path": "data/okapi_length2/email/train.tsv",
    "validation_data_path": "data/okapi_length2/email/dev.tsv",
    "test_data_path": "data/okapi_length2/email/test.tsv",
    "trainer": {
        "callbacks": [
            {
                "type": "track_epoch_callback"
            },
            {
                "type": "test_mode"
            }
        ],
        "grad_norm": 3,
        "num_epochs": 80,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "+freq_acc"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 48,
            "sorting_keys": [
                "source_tokens"
            ]
        }
    },
    "evaluate_on_test": true,
    "numpy_seed": 9529,
    "pytorch_seed": 9530,
    "random_seed": 9528,
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 48,
            "sorting_keys": [
                "source_tokens"
            ]
        }
    }
}