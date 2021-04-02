import random
import string
from uuid import UUID


def get_random_string(l=4):
    return "".join(random.choice(string.ascii_lowercase) for _ in range(l))


def get_uuid_version(uuid):
    try:
        return UUID(uuid.decode("utf-8")).version
    except ValueError:
        return None
