import fire
import logging
import solarnet
from solarnet.run import RunTask

if __name__ == '__main__':
    logger = logging.getLogger(solarnet.__name__)
    logger.setLevel(logging.INFO)
    fire.Fire(RunTask)
