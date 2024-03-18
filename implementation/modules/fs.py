import tensorflow as tf

class FuzzySystemLayer(tf.keras.layers.Layer):
    """
    Custom TensorFlow layer representing a fuzzy system for regression tasks.
    This layer will use pre-determined cluster centers and learnable parameters 
    to perform fuzzy logic-based regression.
    """
    def __init__(self, cluster_centers, output_dimension=1, initializer_stddev=0.05, **kwargs):
        super(FuzzySystemLayer, self).__init__(**kwargs)
        self.cluster_centers = tf.Variable(initial_value=cluster_centers,
                                           dtype='float32',
                                           trainable=True,
                                           name='cluster_centers')
        self.output_dimension = output_dimension
        # Use the initializer_stddev parameter for weight initializers
        self.initializer_stddev = initializer_stddev
        self.widths = self.add_weight(name='widths',
                                      shape=(cluster_centers.shape[0],),
                                      initializer=tf.initializers.RandomNormal(mean=1.0, stddev=self.initializer_stddev),
                                      trainable=True,
                                      dtype='float32')
        self.rule_weights = self.add_weight(name='rule_weights',
                                            shape=(cluster_centers.shape[0], output_dimension),
                                            initializer=tf.initializers.RandomNormal(mean=0.0, stddev=self.initializer_stddev),
                                            trainable=True,
                                            dtype='float32')


    def call(self, inputs):
        # Calculate the membership values for each input
        dists_squared = tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.cluster_centers), axis=2)
        memb_vals = tf.exp(-dists_squared / (2 * tf.square(self.widths)))

        # Normalize membership values to sum to 1 across all rules for each sample
        memb_vals_normalized = memb_vals / tf.reduce_sum(memb_vals, axis=1, keepdims=True)

        # Apply rule weights to compute the weighted sum of rule outputs
        weighted_outputs = tf.matmul(memb_vals_normalized, self.rule_weights)

        return weighted_outputs

    def get_config(self):
        config = super(FuzzySystemLayer, self).get_config()
        config.update({
            'cluster_centers': self.cluster_centers.numpy(),
            'output_dimension': self.output_dimension
        })
        return config


def create_fuzzy_model(input_dim, cluster_centers, output_dim=1, learning_rate=0.001, initializer_stddev=0.05):
    inputs = tf.keras.Input(shape=(input_dim,))
    fuzzy_layer = FuzzySystemLayer(cluster_centers=cluster_centers, output_dimension=output_dim, initializer_stddev=initializer_stddev)
    outputs = fuzzy_layer(inputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model



# Usage example
# Assuming cluster_centers are defined and data is prepared
# model = create_fuzzy_model(input_dim=X_train_scaled.shape[1], cluster_centers=cluster_centers)
# model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, validation_split=0.2)
