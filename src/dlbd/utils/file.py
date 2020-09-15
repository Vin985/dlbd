from pathlib import Path


def list_files(path, extensions=None, recursive=False):
    files = []
    extensions = extensions or []
    for item in Path(path).iterdir():
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

