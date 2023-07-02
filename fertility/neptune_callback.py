import os
import subprocess
from typing import Any, Dict, List, Tuple, Optional
from zipfile import ZipFile

import sys

import numpy as np
from allennlp.training import TrainerCallback

try:
    from gitinfo import get_git_info
except ImportError:
    def get_git_info():
        return dict()

import json

def flatten(d : Dict[Any, Any]):
    """
    Flattens a dictionary and uses the path separated with _ to give unique key names.
    :param d:
    :return:
    """
    r = dict()
    agenda : List[Tuple[Any, List, Any]] = [ (key,[],d) for key in d.keys()]
    while agenda:
        key,path,d = agenda.pop()
        if not isinstance(d[key],dict):
            r["_".join(path+[str(key)])] = d[key]
        else:
            for subkey in d[key].keys():
                agenda.append((subkey,path+[str(key)],d[key]))
    return r


@TrainerCallback.register("neptune")
class NeptuneCallBack(TrainerCallback):
    """
    Writes serialization dir into model
    """

    DO_NOT_LOG = {"training_epoch", "training_start_epoch", "epoch"}

    def __init__(self, serialization_dir: str, project_name:str, experiment_name: Optional[str] = None,
                 offline: bool = False, tags: List[str] = None):
        super().__init__(serialization_dir)
        import neptune.new as neptune

        if experiment_name is None:
            experiment_name = serialization_dir


        with open(os.path.join(serialization_dir, "config.json")) as f:
            config = json.load(f)

        #config = flatten(config)
        config["SERIALIZATION_DIR"] = serialization_dir

        self.experiment_name = experiment_name
        self.project_name = project_name

        tags = tags or []

        if offline:
            self.experiment = neptune.init(project_name, capture_stdout=False, capture_stderr=False, name=experiment_name,
                                                    source_files=[], mode="offline", tags=tags)
            self.run_id = None
        else:
            self.experiment = neptune.init(project_name, capture_stdout=False, capture_stderr=False, name=experiment_name,
                                                        source_files=[], tags=tags)

            self.run_id = self.experiment["sys/id"].fetch()
        self.experiment["parameters"] = config
        self.experiment["config/config.json"].upload(os.path.abspath(os.path.join(serialization_dir, "config.json")))
        self.experiment["config/git-info"] = get_git_info()
        self.experiment["config/argv"] = sys.argv

    def _should_log_metric(self, name:str) -> bool:
        if name in NeptuneCallBack.DO_NOT_LOG:
            return False

        return True

    def log_metric(self, raw_name:str, value:Any):
        if "validation" in raw_name:
            new_name = "dev/" + raw_name.replace("validation", "").replace("__", "_").strip("_")
        elif "training" in raw_name:
            new_name = "train/" + raw_name.replace("training", "").replace("__", "_").strip("_")
        elif "test" in raw_name:
            new_name = "test/" + raw_name.replace("test", "").replace("__", "_").strip("_")
        else:
            new_name = "other_metrics/" + raw_name.strip("_")

        self.experiment[new_name].log(value)

    def on_epoch(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any],
        epoch: int,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        self.last_metrics = dict(metrics)
        for k,v in metrics.items():
            if self._should_log_metric(k) and type(v) is int or type(v) is float or np.isscalar(v):
                self.log_metric(k,v)


    def on_end(
        self,
        trainer: "GradientDescentTrainer",
        metrics: Dict[str, Any] = None,
        epoch: int = None,
        is_primary: bool = True,
        **kwargs,
    ) -> None:
        for k,v in metrics.items():
            if self._should_log_metric(k) and type(v) is int or type(v) is float:
                if self.last_metrics[k] != v:
                    #only log things that have changed because on_epoch might have been called just seconds before.
                    self.log_metric(k, v)

    def __del__(self):
        """
        Neptune callback is destroyed, maybe training ended? Upload any metrics put into metrics.json.
        :return:
        """
        # This needs to be a new process, otherwise it won't work, so it's a bit of a hack.

        if self.run_id is not None:
            f = f'import os,json; import neptune.new as neptune; f = open(os.path.join("{self.serialization_dir}", "metrics.json")); metrics = json.load(f);' \
                f' exp = neptune.init("{self.project_name}", run="{self.run_id}"); ' \
                f'a = [exp["test/"+k].log(v) for k,v in metrics.items() if ("test" in k and (type(v) is int or type(v) is float))];'

            subprocess.run(["python", "-c", f])


