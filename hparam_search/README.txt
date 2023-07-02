you can run hyperparameter optimization like this (here for the calendar domain):

for a hyperparameter search, copy a *space.json and a matching *.jsonnet file to the configs/ directory and then run something like this:

allentune search --experiment-name [experiment_name] --num-cpus 2 --num-gpus 1  --cpus-per-trial 2  --gpus-per-trial 1  --search-space configs/multiset_space.json --num-samples 20 --base-config configs/multiset_okapi_search_calendar.jsonnet --include-package my_package

