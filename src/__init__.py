import logging
import warnings

from matplotlib import pyplot as plt
from rich.logging import RichHandler

# close numba warnings
warnings.simplefilter('ignore')

# close annoying tensorflow warn
logger = logging.getLogger("tensorflow")
logger.setLevel(logging.ERROR)
logger = logging.getLogger("pm2.5 deep learn")
formatter = logging.Formatter("%(name)s :: %(message)s", datefmt="%m-%d %H:%M:%S")

# stdout handler use RichHandler
handler = RichHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

# change useful logger handler
for logger_name in ["tensorflow msg"]:
    _logger = logging.getLogger(logger_name)
    for _handler in _logger.handlers:
        _logger.removeHandler(_handler)
    _logger.addHandler(handler)

# restrict memory usage
# gpus = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=22000)])

# matplotlib setting
plt.rcParams['font.size'] = 16
plt.rc('font', family='Times New Roman')


__all__ = ["logger"]
