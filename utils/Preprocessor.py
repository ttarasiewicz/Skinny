import tensorflow as tf


class Preprocessor:
    def __init__(self):
        self.operations = []

    def downscale(self, max_pixel_count):
        def downscale_operation(data):
            for k, v in data.items():
                tensor_shape = tf.cast(tf.shape(v), tf.float32)
                coefficient = max_pixel_count / (tensor_shape[0] * tensor_shape[1])
                coefficient = tf.math.sqrt(coefficient)
                data[k] = tf.cond(coefficient >= 1.0, lambda: v,
                                  lambda: tf.image.resize(v, [tf.cast(tensor_shape[0] * coefficient, tf.uint16),
                                                              tf.cast(tensor_shape[1] * coefficient, tf.uint16)]))
            return data

        self.operations.append(downscale_operation)
        return self

    def cast(self, dtype):
        def cast_operation(data):
            for k, v in data.items():
                data[k] = tf.cast(v, dtype)
            return data

        self.operations.append(cast_operation)
        return self

    def normalize(self):
        def normalize_operation(data):
            for k, v in data.items():
                data[k] = v / 255.0
            return data

        self.operations.append(normalize_operation)
        return self

    def pad(self, network_levels):
        number_multiple = 2**(network_levels-1)

        def padding_operation(data):
            for k, v in data.items():
                tensor_shape = tf.shape(v)
                data[k] = tf.pad(v, [[0, number_multiple - tensor_shape[0] % number_multiple],
                                     [0,  number_multiple - tensor_shape[1] % number_multiple],
                                     [0, 0]])
            return data

        self.operations.append(padding_operation)
        return self

    def add_to_graph(self, dataset) -> tf.data.Dataset:
        for operation in self.operations:
            dataset = dataset.map(operation)
        return dataset
