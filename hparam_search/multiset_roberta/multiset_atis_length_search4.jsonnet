local embed_dim = 768;

local epochs = 40; 

local batch_size = 48;

local seed=std.parseJson(std.extVar("seed"));

local pretrain_epochs = std.parseJson(std.extVar("pretrain_epochs"));

local thresh = std.parseJson(std.extVar("thresh"));

local weight = std.parseJson(std.extVar("weight"));

local mlp_dim = std.parseJson(std.extVar("mlp_dim"));

local dropout = std.parseJson(std.extVar("dropout"));

local lr =  std.parseJson(std.extVar("lr"));

local neptune = "multiset-atis-search4";

local model = "roberta-base"; #std.extVar("plm");

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


	"source_token_indexers": {
            "bert_tokens": {
                "type": "pretrained_transformer_mismatched",
                "model_name": model
            }
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

        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "bert_tokens": {
                    "type": "pretrained_transformer_mismatched_rho",
                    "rho": 1.0,
                    "train_rho": false,
                    "model_name": model,
                    "train_parameters": true,
                    "freeze_embeddings": false
                },
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

         "parameter_groups": [
              [["source_embedder.*"], {"lr": lr}]
         ]

        },

        "callbacks": [{"type": "track_epoch_callback"}, {"type": "neptune", "project_name": neptune}, {"type": "test_mode"}],

        "validation_metric": "+freq_acc",

    }
}
