{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "copy": true,
        "em_epochs": 150,
        "enable_cogs_var": true,
        "end_symbol": "</s>",
        "lexicon": {
            "type": "copy_lexicon"
        },
        "preprocessor": {
            "type": "aggressive_cogs",
            "simplify": false
        },
        "source_add_end_token": false,
        "source_add_start_token": false,
        "source_token_indexers": {
            "bert_tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "roberta-base"
            }
        },
        "source_tokenizer": {
            "type": "whitespace"
        },
        "start_symbol": "<s>",
        "target_tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "multiset_pred",
        "alignment_loss_weight": 1,
        "lexicon": {
            "type": "copy_lexicon"
        },
        "max_n": 4,
        "mlp": {
            "activations": "gelu",
            "dropout": 0.2,
            "hidden_dims": 512,
            "input_dim": 768,
            "num_layers": 1
        },
        "pretrain_epochs": 10,
        "rho": 0.05,
        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "bert_tokens": {
                    "type": "pretrained_transformer_mismatched_rho",
                    "model_name": "roberta-base",
                    "rho": 1,
                    "train_parameters": true,
                    "train_rho": false
                }
            }
        }
    },
    "train_data_path": "data/COGS/data/train.tsv",
    "validation_data_path": "data/COGS/data/dev.tsv",
    "test_data_path": "data/COGS/data/gen.tsv",
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
            "lr": 0.001,
            "parameter_groups": [
                [
                    [
                        "source_embedder.*"
                    ],
                    {
                        "lr": 1e-05
                    }
                ]
            ]
        },
        "validation_metric": "-loss"
    },
    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
            "sorting_keys": [
                "source_tokens"
            ]
        }
    },
    "evaluate_on_test": true,
    "numpy_seed": 6872,
    "pytorch_seed": 6873,
    "random_seed": 6871,
    "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": 64,
            "sorting_keys": [
                "source_tokens"
            ]
        }
    }
}