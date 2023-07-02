import allennlp
from allennlp.models import load_archive

import allennlp.common.util as common_util

common_util.import_module_and_submodules("fertility")
import sys

if __name__ == "__main__":

    model = load_archive(sys.argv[1]).model

    model.train(True)
    num_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    print("Archive", sys.argv[1])
    print("Number of parameters:", num_params)
    print("Number of parameters in millions:", round(num_params / 1e6,3))

