{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align",
        "copy": false,
        "em_epochs": 150,
        "end_symbol": "</s>",
        "source_add_end_token": true,
        "source_add_start_token": false,
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
            "hidden_size": 100,
            "input_size": 200,
            "layer_dropout_probability": 0.2,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.2
        },
        "metrics": [
            {
                "type": "atis_acc"
            },
            {
                "type": "length_error"
            }
        ],
        "multiset_model_path": std.extVar("model_file"),
        "permutation_model": {
            "type": "var_bregman_bla",
            "encoder_dim": 464,
            "geometric_attention": true,
            "hidden_dim": 512,
            "inference_temp": 10,
            "max_iter": 3000,
            "mlp": {
                "activations": "gelu",
                "dropout": 0.1,
                "hidden_dims": 200,
                "input_dim": 464,
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
                    "embedding_dim": 200,
                    "pretrained_file": "~/downloads/glove.6B.200d.txt",
                    "trainable": true
                }
            }
        },
        "target_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 200,
                    "pretrained_file": "~/downloads/glove.6B.200d.txt",
                    "trainable": true,
                    "vocab_namespace": "target_tokens"
                }
            }
        }
    },
    "train_data_path": "data/atis/atis_funql_train_brackets.tsv",
    "validation_data_path": "data/atis/atis_funql_dev_brackets.tsv",
    "test_data_path": "data/atis/atis_funql_test_brackets.tsv",
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
        "validation_metric": "+tree_acc"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 32,
            "sorting_keys": [
                "target_tokens"
            ]
        }
    },
    "evaluate_on_test": true,
    "numpy_seed": 7892,
    "pytorch_seed": 7893,
    "random_seed": 7891,
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