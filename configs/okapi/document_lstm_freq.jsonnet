{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "em_epochs": 150,
        "end_symbol": "</s>",
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/document/train.tsv",
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
        "alignment_loss_weight": 0.955622,
        "alignment_threshold": 0.806403,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 100,
            "input_size": 200,
            "layer_dropout_probability": 0.369286,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.369286
        },
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/document/train.tsv",
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
            "dropout": 0.223524,
            "hidden_dims": 256,
            "input_dim": 200,
            "num_layers": 1
        },
        "pretrain_epochs": 6,
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
    "train_data_path": "data/okapi_length2/document/train.tsv",
    "validation_data_path": "data/okapi_length2/document/dev.tsv",
    "test_data_path": "data/okapi_length2/document/test.tsv",
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
    "numpy_seed": 7225,
    "pytorch_seed": 7226,
    "random_seed": 7224,
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