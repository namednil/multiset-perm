local embed_dim = 768;

local epochs = 40; # set to 60

local batch_size = 48; # TODO: set to 48

local seed=std.parseJson(std.extVar("seed"));

local pretrain_epochs = std.parseJson(std.extVar("pretrain_epochs"));

local thresh = std.parseJson(std.extVar("thresh"));

local weight = std.parseJson(std.extVar("weight"));

local mlp_dim = std.parseJson(std.extVar("mlp_dim"));

local dropout = std.parseJson(std.extVar("dropout"));

local lr =  std.parseJson(std.extVar("lr"));


//local domain = std.extVar("domain");
local domain = "email";

local neptune = "multiset-okapi-search";

local model = "roberta-base";

local train_data = "/home/ID/data/okapi_length2/"+domain+"/train.tsv";
local dev_data =  "/home/ID/data/okapi_length2/"+domain+"/dev.tsv";
local test_data = "/home/ID/data/okapi_length2/"+domain+"/test.tsv";


local lexicon = {
            "type": "simple_lexicon",
            "copy": true,

            "filename": train_data,

            "copy_despite_case_mismatch": true,

            # Following Herzig & Berant, predicates cannot be copied, so we are making a list of exceptions:
            "non_copyable": ["GET", "FILTER", "SEARCH", "eq", "gt", "lt", "True", "False", "ORDERBY", "COUNT", "asc", "desc",
                        "TOP"],

            "source_tokenizer": {
                "type": "spacy"
            },

            "target_tokenizer": {
                "type": "whitespace"
            },
};

local dataset_reader = {
        "type": "tsv_reader_with_copy_align_counter",

        "em_epochs": 150,

        "lexicon": lexicon,

        "source_add_start_token": false,
        "source_add_end_token": true,

       "start_symbol": "<s>", # seem to be added under the hood already!
       "end_symbol": "</s>",


        "source_tokenizer": {
            "type": "spacy"
        },

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


    "train_data_path": train_data,
    "validation_data_path": dev_data,

    "dataset_reader": dataset_reader,

    "model": {
        "type": "multiset_pred",

        "pretrain_epochs": pretrain_epochs,

        "alignment_loss_weight": weight,

        "alignment_threshold": thresh,

        "lower_case_eval": true,

        "lexicon": lexicon,

        "max_n": 4,

        "source_embedder": {
            "type": "basic",
            "token_embedders": {
                "bert_tokens": {
                    "type": "pretrained_transformer_mismatched_rho",
                    "rho": 1.0,
                    "train_rho": false,
                    "model_name": model,
                    "train_parameters": true
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

//        "callbacks": [{"type": "track_epoch_callback"}, {"type": "test_mode"}],
        "callbacks": [{"type": "track_epoch_callback"}, {"type": "neptune", "project_name": neptune}, {"type": "test_mode"}],

        "validation_metric": "+freq_acc",

    }
}

