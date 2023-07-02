{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "copy": false,
        "em_epochs": 150,
        "end_symbol": "</s>",
        "source_add_end_token": true,
        "source_add_start_token": false,
        "source_token_indexers": {
            "bert_tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "roberta-base"
            }
        },
        "start_symbol": "<s>",
        "target_tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "multiset_pred",
        "alignment_loss_weight": 0.808971,
        "alignment_threshold": 0.604612,
        "max_n": 4,
        "mlp": {
            "activations": "gelu",
            "dropout": 0.339884,
            "hidden_dims": 256,
            "input_dim": 768,
            "num_layers": 1
        },
        "pretrain_epochs": 5,
        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "bert_tokens": {
                    "type": "pretrained_transformer_mismatched_rho",
                    "freeze_embeddings": false,
                    "model_name": "roberta-base",
                    "rho": 1,
                    "train_parameters": true,
                    "train_rho": false
                }
            }
        }
    },
    "train_data_path": "data/atis/atis_funql_length4_train_brackets.tsv",
    "validation_data_path": "data/atis/atis_funql_length4_dev_brackets.tsv",
    "test_data_path": "data/atis/atis_funql_length4_test_brackets.tsv",
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
        "num_epochs": 60,
        "optimizer": {
            "type": "adam",
            "lr": 0.001,
            "parameter_groups": [
                [
                    [
                        "source_embedder.*"
                    ],
                    {
                        "lr": 6.762542674692122e-06
                    }
                ]
            ]
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
    "numpy_seed": 7087,
    "pytorch_seed": 7088,
    "random_seed": 7086,
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