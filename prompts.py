from __future__ import annotations

from typing import Dict, List, Tuple


TASK_TEMPLATES = [
    "CLASSNAME.",
    "a histopathological whole slide image of CLASSNAME.",
    "an H&E stained whole slide image of CLASSNAME.",
    "a pathology image from CLASSNAME.",
]


TASK_NAMES: Dict[str, List[str]] = {
    "blca": ["bladder urothelial carcinoma", "bladder cancer", "TCGA-BLCA"],
    "brca": ["breast invasive carcinoma", "breast cancer", "TCGA-BRCA"],
    "luad": ["lung adenocarcinoma", "lung cancer adenocarcinoma", "TCGA-LUAD"],
    "ucec": ["uterine corpus endometrial carcinoma", "endometrial cancer", "TCGA-UCEC"],
}


def task_level_prompts(task: str) -> Tuple[List[List[str]], List[str]]:
    """Task-level prompts for CLASS-IL routing after merging."""
    if task not in TASK_NAMES:
        raise KeyError(f"Unknown task {task!r}. Available: {sorted(TASK_NAMES)}")
    return [[name for name in TASK_NAMES[task]]], TASK_TEMPLATES
