local embed_dim = 200;

local epochs = 40; 

local batch_size = 48;

local seed=std.parseJson(std.extVar("seed"));

local pretrain_epochs = std.parseJson(std.extVar("pretrain_epochs"));

local thresh = std.parseJson(std.extVar("thresh"));

local weight = std.parseJson(std.extVar("weight"));

local mlp_dim = std.parseJson(std.extVar("mlp_dim"));

local dropout = std.parseJson(std.extVar("dropout"));

local lstm_dropout = std.parseJson(std.extVar("lstm_dropout"));

####local lr =  std.parseJson(std.extVar("lr"));

local neptune = "multiset-atis-search-lstm";


local dataset_reader = {
        "type": "tsv_reader_with_copy_align_counter",

        "em_epochs": 150,

        "copy": false,

        "source_add_start_token": false,
        "source_add_end_token": true,

       "start_symbol": "<s>", # seem to be added under the hood already!
       "end_symbol": "</s>",


        "target_tokenizer": {
            "type": "whitespace"
        },



};

{

    "random_seed": seed,
    "numpy_seed": seed+1,
    "pytorch_seed": seed+2,

    "train_data_path": "/home/ID/data/atis/atis_funql_length4_train_brackets.tsv",

    "validation_data_path": "/home/ID/data/atis/atis_funql_length4_dev_brackets.tsv",

    "evaluate_on_test": false,

    "dataset_reader": dataset_reader,

    "model": {
        "type": "multiset_pred",

        "pretrain_epochs": pretrain_epochs,

        "alignment_loss_weight": weight,

        "alignment_threshold": thresh, #was 0.8

        "max_n": 4,


        "rho": 1.0,

         "encoder": {
            "type": "stacked_bidirectional_lstm",
            "input_size": embed_dim,
            "hidden_size": embed_dim / 2,
            "num_layers": 3,
            "recurrent_dropout_probability": lstm_dropout,
            "layer_dropout_probability": lstm_dropout,
        },


        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "tokens": {
                    "type" : "embedding",
                    "embedding_dim" : embed_dim,
                    "pretrained_file": "~/downloads/glove.6B.200d.txt",
                    "trainable": true
                    }
            }
        },


        "mlp": {
            "input_dim": embed_dim, 
            "hidden_dims": mlp_dim,
            "num_layers": 1,
            "activations": "gelu",
            "dropout": dropout
        },


    },



    "data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": ["source_tokens"]
        }
    },

   "validation_data_loader": {
        "batch_sampler": {
            "type": "bucket",
            "batch_size": batch_size,
            "sorting_keys": ["source_tokens"]
        }
    },

    "trainer": {
        "num_epochs": epochs,

        "grad_norm": 3,

        "optimizer": {
            "type": "adam",
            "lr": 1e-03,

        },

        "callbacks": [{"type": "track_epoch_callback"}, {"type": "neptune", "project_name": neptune}, {"type": "test_mode"}],

        "validation_metric": "+freq_acc",

    }
}
