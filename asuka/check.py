import importlib.util

from loguru import logger


def check_package_installed(package_name: str) -> bool:
    spec = importlib.util.find_spec(package_name)
    if spec is not None:
        return True
    else:
        logger.warning(f"{package_name} is not installed, will ignore this package.")
        return False
