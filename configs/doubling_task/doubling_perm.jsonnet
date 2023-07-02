{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align",
        "copy": false,
        "em_epochs": 150,
        "end_symbol": "</s>",
        "source_add_end_token": true,
        "source_add_start_token": false,
        "source_tokenizer": {
            "type": "whitespace"
        },
        "start_symbol": "<s>",
        "target_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "target_tokens"
            }
        },
        "target_tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "multiset_reorder",
        "concat_reprs": true,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 48,
            "input_size": 96,
            "layer_dropout_probability": 0.2,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.2
        },
        "metrics": [
            {
                "type": "levenshtein"
            },
            {
                "type": "length_error"
            },
            {
                "type": "acc_by_length"
            }
        ],
        "multiset_model_path": std.extVar("model_file"),
        "permutation_model": {
            "type": "var_bregman_bla",
            "encoder_dim": 256,
            "geometric_attention": true,
            "hidden_dim": 512,
            "inference_temp": 10,
            "max_iter": 3000,
            "mlp": {
                "activations": "gelu",
                "dropout": 0.1,
                "hidden_dims": 96,
                "input_dim": 256,
                "num_layers": 1
            },
            "namespace": "target_tokens",
            "omega": 1,
            "posterior_max_iter": 150,
            "postprocess_with_lap": true,
            "rel_dot": false,
            "tol": 1e-06,
            "train_max_iter": 150
        },
        "positional_embedding_dim": 64,
        "rho": 1,
        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 96,
                    "trainable": true
                }
            }
        },
        "target_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 96,
                    "trainable": true,
                    "vocab_namespace": "target_tokens"
                }
            }
        }
    },
    "train_data_path": "data/copy3/train.tsv",
    "validation_data_path": "data/copy3/dev.tsv",
    "test_data_path": "data/copy3/test.tsv",
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
        "num_epochs": 20,
        "optimizer": {
            "type": "adam",
            "lr": 0.001
        },
        "validation_metric": "-loss"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 48,
            "sorting_keys": [
                "target_tokens"
            ]
        }
    },
    "evaluate_on_test": true,
    "numpy_seed": 8229,
    "pytorch_seed": 8230,
    "random_seed": 8228,
    "validation_data_loader": {
        "batch_sampler": {
            "type": "validation_bucket_length",
            "batch_sizes": [
                [
                    0,
                    10,
                    128
                ],
                [
                    10,
                    30,
                    64
                ],
                [
                    30,
                    40,
                    32
                ],
                [
                    40,
                    200,
                    16
                ]
            ],
            "length_key": "target_tokens"
        }
    }
}
