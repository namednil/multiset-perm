

# Compositional Generalization without Trees using Multiset Tagging and Latent Permutations
This is the official code for our ACL 2023 paper [Compositional Generalization without Trees using Multiset Tagging and Latent Permutations](https://arxiv.org/abs/2305.16954).


## Usage
### Installation
Clone this repository, and unzip `data.zip`. Then create a conda environment with Python 3.8:
```
conda create -n f-r python=3.8
conda activate f-r
```
And install [PyTorch 1.8](https://pytorch.org/get-started/previous-versions/):

```
# CUDA 10.2
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu102

# CUDA 11.1
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

# CPU Only
pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cpu
```
Then install the remaining requirements (this may take a while):
```
pip install -r requirements.txt
```
Place GloVe embeddings into `~/downloads/` or the adapt config files (`configs/`) accordingly to point to where you saved the GloVe embeddings.

### Training a model
A configuration file describes a model, the data you want to train on and all hyperparameters. Pick a configuration file from `configs/`
Ensure that `[train|dev|test]_data_path` point to the right data; also make sure that `pretrained_file` points to your copy of the [GloVe embeddings](https://nlp.stanford.edu/data/glove.6B.zip). Then run
We train two models, one for predicting multisets, and after that the one for predicting permutations.

To train both models, run
```
python train_both.py [path to where model will be saved] [multiset model config] [permutation model config]
```
For example, to train a model for COGS (using the lexicon), run
```
mkdir -p models/
python train_both.py models/multiset_cogs_lexicon_model configs/cogs/cogs_lexicon_freq.jsonnet configs/cogs/cogs_lexicon_perm.jsonnet
```
This will create two directories, `models/multiset_cogs_lexicon_model_freq` and `models/multiset_cogs_lexicon_model_perm` containing the corresponding two models. 
I recommend not moving these directories (otherwise paths must be adjusted).
You can also add the additional argument `stage_1` or `stage_2` to train only one of the models.

If you want the experiment to be logged by neptune.ai, make sure that `trainer.callbacks` contains an entry like this:
``` 
{ "type": "neptune", "project_name": "[project name]" }
```
I you want to tune hyperparameters automatically, you can use [this fork of allentune](https://github.com/namednil/allentune). Files with search spaces for hyperparameters are in `hparam_search/`.

### Evaluation of a model
You can compute **evaluation** metrics on some data like this:
```
allennlp evaluate [path/to/model.tar.gz] [path/to/data.tsv] --include-package fertility
```
Be sure to use a model trained for predicting permutations (ending in `_perm`) to get evaluation results on sequence outputs (and not multisets).
If you want to use a cuda device, add `--cuda-device [device_id]` to that. 
See `allennlp evaluate --help` for more options such as saving
the metrics without rounding.

If you want to save predictions of the model on same data (e.g. for error analysis), use this:
```
allennlp predict [path/to/model.tar.gz] [path/to/data.tsv] --include-package fertility --use-dataset-reader --output-file [outout-file.jsonl]
```

## Inference for Relaxed Permutations
We introduce a generalization of the Sinkhorn-Knopp algorithm for inference with relaxed permutations that are parameterized with jumps.

You can find the code for this algorithm in `fertility/reordering/bregman_for_perm.py`, which is a standalone file that
also includes a simple example. Feel free to reach out if you have any questions!

## Citation

```
@inproceedings{lindemann-etal-2023a-compositional,
    title = "Compositional Generalization without Trees using Multiset Tagging and Latent Permutations",
    author = "Lindemann, Matthias  and
      Koller, Alexander  and
      Titov, Ivan",
    booktitle = "Proceedings of the 61st Conference of the ACL",
    month = july,
    year = "2023",
    address = "Toronto, Canada",
    url = "https://arxiv.org/abs/2305.16954",
}
```

