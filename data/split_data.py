from pathlib import Path
from random import Random
from shutil import copy2, rmtree


SEED = 42
BASE_DIR = Path(__file__).resolve().parent.parent
SOURCE_DIR = BASE_DIR / "evaluation_results" / "txt_files_function_calling_evaluation"
PROMPT_DIR = BASE_DIR / "evaluation_results" / "txt_files_prompt_selection"
MAIN_DIR = BASE_DIR / "evaluation_results" / "txt_files_main_evaluation"
REPEATABILITY_DIR = BASE_DIR / "evaluation_results" / "txt_files_repeatability"
REASONING_DIR = BASE_DIR / "evaluation_results" / "txt_files_reasoning_evaluation"


def recreate_directory(path: Path):
    if path.exists():
        rmtree(path)

    path.mkdir(parents=True)


def copy_files(files, destination):
    for file in files:
        copy2(file, destination / file.name)


def main():
    rng = Random(SEED)
    groups = {"civil_pleas": [], "crown_pleas": [],"foreign_pleas": []}

    for file in SOURCE_DIR.glob("*.txt"):
        for prefix in groups:
            if file.stem.startswith(prefix):
                groups[prefix].append(file)
                break

    for files in groups.values():
        files.sort()
        rng.shuffle(files)

    prompt_files = []
    main_files = []

    for files in groups.values():
        prompt_files.extend(files[:10])
        main_files.extend(files[10:])

    main_groups = {
        prefix: [ file for file in main_files if file.stem.startswith(prefix)] for prefix in groups
    }

    reasoning_rng = Random(SEED + 1)
    reasoning_files = reasoning_rng.sample(sorted(main_files),50,)
    repeatability_files = []

    for files in main_groups.values():
        repeatability_files.extend(rng.sample(files, 10))

    assert len(prompt_files) == 30
    assert len(main_files) == 70
    assert len(repeatability_files) == 30
    assert len(reasoning_files) == 50
    assert {file.name for file in prompt_files}.isdisjoint({file.name for file in main_files})
    assert {file.name for file in prompt_files}.isdisjoint({file.name for file in reasoning_files})

    recreate_directory(PROMPT_DIR)
    recreate_directory(MAIN_DIR)
    recreate_directory(REPEATABILITY_DIR)
    recreate_directory(REASONING_DIR)
    copy_files(prompt_files, PROMPT_DIR)
    copy_files(main_files, MAIN_DIR)
    copy_files(repeatability_files, REPEATABILITY_DIR)
    copy_files(reasoning_files, REASONING_DIR)


if __name__ == "__main__":
    main()