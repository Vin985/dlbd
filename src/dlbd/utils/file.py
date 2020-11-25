from pathlib import Path

import yaml


def ensure_path_exists(path, is_file=False):
    if is_file:
        tmp = path.parent
    else:
        tmp = path
    if not tmp.exists():
        tmp.mkdir(exist_ok=True, parents=True)
    return path


def list_files(path, extensions=None, recursive=False):
    files = []
    extensions = extensions or []
    path = Path(path)
    if not path.exists():
        return files
    for item in path.iterdir():
        if item.is_dir() and recursive:
            files += list_files(item, extensions, recursive)
        elif item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return files


def list_folder(path, extensions=None):
    dirs = []
    files = []
    extensions = extensions or []
    for item in Path(path).iterdir():
        if item.is_dir():
            dirs.append(item)
        elif item.is_file() and item.suffix.lower() in extensions:
            files.append(item)
    return (dirs, files)


def load_config(path):
    stream = open(path, "r")
    config = yaml.load(stream, Loader=yaml.Loader)
    return config


def get_full_path(path, root):
    if path.is_absolute() or not root:
        return path
    else:
        return root / path

