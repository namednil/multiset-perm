local seed = std.parseJson(std.extVar("seed"));

local hid_dim = std.parseJson(std.extVar("hid_dim"));

local batch_size = std.parseJson(std.extVar("batch_size"));

local num_layers = std.parseJson(std.extVar("num_layers"));
local rec_dropout = std.parseJson(std.extVar("rec_dropout"));
local layer_dropout = std.parseJson(std.extVar("layer_dropout"));
local target_decoder_layers = std.parseJson(std.extVar("target_decoder_layers"));

local embed_dim = 200;
local epochs = 20;

local neptune = "baseline-copy3";


{

    "random_seed": seed,
    "numpy_seed": seed+1,
    "pytorch_seed": seed+2,

    "train_data_path": "/home/ID/data/copy3/train.tsv",

    "validation_data_path": "/home/ID/data/copy3/dev.tsv",

    "dataset_reader": {
        "type": "my_seq2seq",

         "source_tokenizer": {
            "type": "whitespace"
        },

        "target_tokenizer": {
            "type": "whitespace"
        },


        "source_token_indexers": {
            "tokens": {
                "type": "single_id",
                "lowercase_tokens": true
            }
        },

        "target_token_indexers": {
            "tokens": {
                "type": "single_id",
                "namespace": "target_tokens" #want to keep the target namespace separate from the source namespace
            }
        }

    },
    "model": {
        "type": "lstm_seq2seq",

        "target_namespace": "target_tokens",

        "metrics": [{"type": "levenshtein"}, {"type": "length_error"}, {"type": "acc_by_length"}],

        "beam_search": {
            "max_steps": 60,
            "beam_size": 5,
        },


        "target_decoder_layers": target_decoder_layers,

        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type" : "embedding",
                    "embedding_dim" : embed_dim
                    }
            }
        },
        "encoder": {
            "type": "stacked_bidirectional_lstm",
            "input_size": embed_dim,
            "hidden_size": hid_dim,
            "num_layers": num_layers,
           "recurrent_dropout_probability": rec_dropout ,
           "layer_dropout_probability": layer_dropout,

        },

        "attention": {
            "type": "additive",
            "vector_dim": 2*hid_dim,
            "matrix_dim": 2*hid_dim
        },

        "use_bleu": false
    },

    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": ["target_tokens"]
        }
    },

    "trainer": {
        "num_epochs": epochs,
        "optimizer": {
            "type": "adam"
        },

        "callbacks": [{"type": "track_epoch_callback"}, {"type": "neptune", "project_name": neptune}, {"type": "test_mode"}],

        "validation_metric": "-loss"
    }
}

