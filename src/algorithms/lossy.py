import numpy as np


def uniform_quantization(data, num_levels):
    data = np.array(data)
    data_min, data_max = np.min(data), np.max(data)
    if data_max == data_min:
        return data, 1.0
    step_size = (data_max - data_min) / num_levels
    quantized_data = np.floor((data - data_min) / step_size) * step_size + data_min
    return quantized_data.astype(np.uint8), step_size
