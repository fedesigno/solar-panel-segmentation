from datetime import datetime
from uuid import uuid4


def current_timestamp() -> str:
    return datetime.strftime(datetime.now(), "%Y-%M-%d-%H-%m")


def generate_id() -> str:
    return str(uuid4())
