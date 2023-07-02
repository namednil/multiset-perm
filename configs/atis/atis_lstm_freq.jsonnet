{
    "dataset_reader": {
        "type": "tsv_reader_with_copy_align_counter",
        "copy": false,
        "em_epochs": 150,
        "end_symbol": "</s>",
        "source_add_end_token": true,
        "source_add_start_token": false,
        "start_symbol": "<s>",
        "target_tokenizer": {
            "type": "whitespace"
        }
    },
    "model": {
        "type": "multiset_pred",
        "alignment_loss_weight": 0.857354,
        "alignment_threshold": 0.555815,
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "hidden_size": 100,
            "input_size": 200,
            "layer_dropout_probability": 0.394957,
            "num_layers": 3,
            "recurrent_dropout_probability": 0.394957
        },
        "max_n": 4,
        "mlp": {
            "activations": "gelu",
            "dropout": 0.479709,
            "hidden_dims": 768,
            "input_dim": 200,
            "num_layers": 1
        },
        "pretrain_epochs": 7,
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
        "num_epochs": 60,
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
    "numpy_seed": 7892,
    "pytorch_seed": 7893,
    "random_seed": 7891,
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