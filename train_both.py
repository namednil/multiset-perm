import os
import json
import time
import sys
import shutil
import subprocess


def run_command(command):
    obj = subprocess.Popen(command)
    while obj.poll() is None:
        time.sleep(2)

    obj.wait()
    return obj.returncode


print(sys.argv)

dir = sys.argv[1]
config_1 = sys.argv[2]
config_2 = sys.argv[3]

if len(sys.argv) == 5:
    only_run = sys.argv[4]
    assert only_run in ["stage_1", "stage_2"]
else:
    only_run = None

command1 = [shutil.which("python"), "-m", "allennlp", "train", config_1, "-s", f"{dir}_freq", "-f", "--include-package",
            "fertility", "--file-friendly-logging"]
command2 = [shutil.which("python"), "-m", "allennlp", "train", config_2, "-s", f"{dir}_reorder", "-f",
            "--include-package", "fertility", "--file-friendly-logging"]

if only_run == "stage_2":
    os.environ["model_file"] = f"{dir}_freq"
    run_command(command2)
else:
    ret_code = run_command(command1)
    if ret_code == 0:
        if not only_run == "stage_1":
            print("First command OK, running second")
            os.environ["model_file"] = f"{dir}_freq"
            run_command(command2)
