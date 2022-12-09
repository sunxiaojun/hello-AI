import os


class Config(object):
    def __init__(self):
        pass

    @staticmethod
    def get_project_dir():
        file_dir = os.path.dirname(__file__)
        return os.path.abspath(os.path.join(file_dir, os.pardir))
