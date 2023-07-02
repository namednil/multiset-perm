{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "copy": false,
        "em_epochs": 150,
        "end_symbol": "</s>",
        "source_add_end_token": true,
        "source_add_start_token": false,
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
        "alignment_loss_weight": 0,
        "alignment_threshold": 0,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 48,
            "input_size": 96,
            "layer_dropout_probability": 0.1,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.1
        },
        "max_n": 2,
        "mlp": {
            "activations": "gelu",
            "dropout": 0.1,
            "hidden_dims": 256,
            "input_dim": 96,
            "num_layers": 1
        },
        "pretrain_epochs": 0,
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
        "num_epochs": 15,
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
                "source_tokens"
            ]
        }
    },
    "evaluate_on_test": true,
    "numpy_seed": 8229,
    "pytorch_seed": 8230,
    "random_seed": 8228,
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
