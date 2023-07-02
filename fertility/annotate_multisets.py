from copy import deepcopy
from collections import Counter

import allennlp
from allennlp.common.util import prepare_environment, import_module_and_submodules
from allennlp.data import DataLoader
from allennlp.models import load_archive

# import fertility.multiset_pred
# import fertility.dataset_readers.tsv_reader_with_copy_align_counter


import sys

if __name__ == "__main__":

    import_module_and_submodules("fertility")

    from fertility.multiset_reorder import linearize_multisets
    
    archive_file = sys.argv[1]
    data_path = sys.argv[2]
    print_all = False
    if len(sys.argv) == 4:
       assert sys.argv[3] == "all"
       print_all = True

    archive = load_archive(
        archive_file,
        cuda_device=0
    )
    config = deepcopy(archive.config)
    prepare_environment(config)
    model = archive.model
    model.eval()

    # Load the evaluation data
    dataset_reader = archive.validation_dataset_reader

    data_loader_params = config.get("validation_data_loader", None)
    if data_loader_params is None:
        data_loader_params = config.get("data_loader")


    data_loader = DataLoader.from_params(
        params=data_loader_params, reader=dataset_reader, data_path=data_path
    )

    data_loader.set_target_device(0)

    data_loader.index_with(model.vocab)

    SEP = " "

    for x in data_loader:
        output = model(**x)
        predicted_multisets = output["predicted_multisets"]  # shape (batch, input_seq_len, vocab)
        linearized_multisets_output, alignment, pred_target_mask, copy_info = linearize_multisets(
            predicted_multisets, output["targets"].shape[1])
        readable_source = [m["source_tokens"] for m in x["metadata"]]
        readable_target = [m["target_tokens"] for m in x["metadata"]]
        readable_linearized_outputs = model.make_sequence_readable(linearized_multisets_output, alignment,
                                                                                 readable_source)
        for s, t, ms, m in zip(readable_source, readable_target, readable_linearized_outputs, x["metadata"]):
            if print_all or Counter(ms) == Counter(t):
                # For fairness, only keep those instances where the multiset model gets the overall multiset right.

                output_str = " ".join(s) + SEP + " ".join(ms) + "\t" + " ".join(t)
                if "category" in m and m["category"] is not None:
                    output_str += "\t" + m["category"]
                print(output_str)
