import git
from multiprocessing import Process, Manager
import os


def process_get_version(path, return_dict):
    try:
        repo = git.Repo(
            path=path,
            search_parent_directories=True)
        return_dict["version"] = (
            repo.head.object.hexsha,
            repo.head.commit.authored_datetime.strftime("%b %d %Y %H:%M:%S"),
            repo.head.commit.author.name,
            repo.head.commit.message)
    except Exception as e:
        return_dict["version"] = {
            "error": str(e),
            "path": path
        }


def get_version(path):
    # apparently leaking resources
    manager = Manager()
    return_dict = manager.dict()
    p = Process(target=process_get_version, args=(path, return_dict,))
    p.start()
    p.join()
    return return_dict["version"]


def get_versions(env_variables):
    return {
        env_variable: get_version(os.environ[env_variable])
        for env_variable in env_variables
    }
