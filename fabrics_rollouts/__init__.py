from importlib.metadata import version
__version__ = version(__name__)

from fabrics_rollouts.scripts.tcp_client.fabrics_client import FabricsClient