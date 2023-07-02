local seed = std.parseJson(std.extVar("seed"));

local head_factor = 2*std.parseJson(std.extVar("head_factor")); # we want to make sure that 2 is a factor of the state dimensionality.

local batch_size = std.parseJson(std.extVar("batch_size"));

local num_layers = std.parseJson(std.extVar("num_layers"));
local dropout = std.parseJson(std.extVar("dropout"));
local feedforward_dim = std.parseJson(std.extVar("feedforward_dim"));
local nheads = std.parseJson(std.extVar("nheads"));
local lr = std.parseJson(std.extVar("lr"));

local warmup = std.parseJson(std.extVar("warmup"));

local embed_dim = 200;
local epochs = 25;

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
        "type": "csordas_transformer",

        "transformer_type": "relative",

        "metrics": [{"type": "levenshtein"}, {"type": "length_error"}, {"type": "acc_by_length"}],

        "state_size": head_factor * nheads,
        "dropout": dropout,
        "max_len": 80,
        "dim_feedforward": feedforward_dim,

        "nhead": nheads,
        "num_encoder_layers": num_layers,
        "num_decoder_layers": num_layers,

        #Recommended by Csordás et al.:
        "scale_mode": "down",
        "embedding_init": "kaiming"
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
            "type": "adam",
            "lr": lr,
        },            
            
            "learning_rate_scheduler": {
               "type": "linear_with_warmup",
              "warmup_steps":warmup
           },

        "callbacks": [{"type": "track_epoch_callback"}, {"type": "neptune", "project_name": neptune}, {"type": "test_mode"}],

        "validation_metric": "+seq_acc"
    }
}


