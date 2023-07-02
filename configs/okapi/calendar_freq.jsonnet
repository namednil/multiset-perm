{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "em_epochs": 150,
        "end_symbol": "</s>",
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/calendar/train.tsv",
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
        "source_token_indexers": {
            "bert_tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": "roberta-base"
            }
        },
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
        "alignment_loss_weight": 0.416918,
        "alignment_threshold": 0.639808,
        "lexicon": {
            "type": "simple_lexicon",
            "copy": true,
            "copy_despite_case_mismatch": true,
            "filename": "data/okapi_length2/calendar/train.tsv",
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
            "dropout": 0.201176,
            "hidden_dims": 256,
            "input_dim": 768,
            "num_layers": 1
        },
        "pretrain_epochs": 7,
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
    "train_data_path": "data/okapi_length2/calendar/train.tsv",
    "validation_data_path": "data/okapi_length2/calendar/dev.tsv",
    "test_data_path": "data/okapi_length2/calendar/test.tsv",
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
                        "lr": 2.601468166583931e-05
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
    "numpy_seed": 217,
    "pytorch_seed": 218,
    "random_seed": 216,
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
