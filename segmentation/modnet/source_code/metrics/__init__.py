from .metric import MSE, MAD, Grad, Conn
from .stream_metrics import StreamSegMetrics

metrics_class_dict = {'mad': MAD, 'mse': MSE, 'grad': Grad, 'conn': Conn}
