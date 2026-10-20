from merging.opcm import merge as opcm_merge
from merging.weight_average import merge as weight_average_merge
from merging.task_arithmetic import merge as task_arithmetic_merge
from merging.ties import merge as ties_merge
from merging.dop import merge as dop_merge
from merging.nufilt import merge as nufilt_merge
from merging.dare import merge as dare_merge
from merging.opcm_original import merge as opcm_original_merge

MERGERS = {
    "opcm": opcm_merge,
    "weight_average": weight_average_merge,
    "task_arithmetic": task_arithmetic_merge,
    "ties": ties_merge,
    "dop": dop_merge,
    "nufilt": nufilt_merge,
    "dare": dare_merge,
    "opcm_original": opcm_original_merge,
}

def get_merger(name):
    return MERGERS[name]