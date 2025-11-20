from os import makedirs
from os.path import exists


def check_path(*args):
    """
    check paths and create them
    Args:
        *args: path lists
    """
    for _path in args:
        if not exists(_path):
            makedirs(_path)

