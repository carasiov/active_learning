import numpy as np
import keras
import tensorflow as tf
import tensorflow_probability as tfp
from keras.callbacks import EarlyStopping, ModelCheckpoint  # type: ignore
import matplotlib.pyplot as plt
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.val_losses = []
        self.reconstruction_loss = []
        self.val_reconstruction_loss = []
        self.classification_loss = []
        self.val_classification_loss = []
        self.kl_loss = []
        self.val_kl_loss = []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.reconstruction_loss.append(logs.get('reconstruction_loss'))
        self.val_reconstruction_loss.append(logs.get('val_reconstruction_loss'))
        self.classification_loss.append(logs.get('classification_loss'))
        self.val_classification_loss.append(logs.get('val_classification_loss'))
        self.kl_loss.append(logs.get('kl_loss'))
        self.val_kl_loss.append(logs.get('val_kl_loss'))
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        axes[0].plot(self.losses, label='Training Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        ax2 = axes[1].twinx()
        axes[1].plot(self.reconstruction_loss, label='Reconstruction Loss', color='blue')
        axes[1].plot(self.val_reconstruction_loss, label='Validation Reconstruction Loss', color='cyan')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction and Classification Losses')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)
        ax2.plot(self.kl_loss, label='KL Loss', color='red')
        ax2.plot(self.val_kl_loss, label='Validation KL Loss', color='orange')
        ax2.legend(loc='upper right')
        ax3 = ax2.twinx()
        ax3.plot(self.classification_loss, label='Classification Loss', color='green')
        ax3.plot(self.val_classification_loss, label='Validation Classification Loss', color='lime')
        ax3.legend()
        plt.tight_layout()
        plt.savefig('models/progress/ssvae_loss_plot.png')
        plt.close()
class Sampling(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dist = tfp.distributions.Normal(loc=0.0, scale=1.0)
    def call(self, inputs, training=None):
        mu, log_var = inputs
        epsilon = self.dist.sample(tf.shape(mu))
        return mu + tf.exp(0.5 * log_var) * epsilon
class SemiSupervisedVariationalAutoEncoderKerasModel(keras.Model):
    def __init__(self, encoder, decoder, classifier, **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        self.total_loss_tracker = keras.metrics.Mean(name='total_loss')
        self.reconstruction_loss_tracker = keras.metrics.Mean(name='reconstruction_loss')
        self.kl_loss_tracker = keras.metrics.Mean(name='kl_loss')
        self.classification_loss_tracker = keras.metrics.Mean(name='classification_loss')
        self.contrastive_loss_tracker = keras.metrics.Mean(name='contrastive_loss')
    def get_config(self):
        config = super().get_config()
        config.update({
            'encoder': keras.saving.serialize_keras_object(self.encoder),
            'decoder': keras.saving.serialize_keras_object(self.decoder),
            'classifier': keras.saving.serialize_keras_object(self.classifier),
        })
        return config
    @classmethod
    def from_config(cls, config):
        encoder = keras.saving.deserialize_keras_object(config.pop('encoder'))
        decoder = keras.saving.deserialize_keras_object(config.pop('decoder'))
        classifier = keras.saving.deserialize_keras_object(config.pop('classifier'))
        return cls(encoder=encoder, decoder=decoder, classifier=classifier, **config)
    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z_mean), self.classifier(z_mean)
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.classification_loss_tracker,
            self.contrastive_loss_tracker,
        ]
    def train_step(self, data):
        total_loss, tape = self.calculate_loss(data)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'classification_loss': self.classification_loss_tracker.result(),
            'contrastive_loss': self.contrastive_loss_tracker.result(),
        }
    def test_step(self, data):
        self.calculate_loss(data)
        return {
            'loss': self.total_loss_tracker.result(),
            'reconstruction_loss': self.reconstruction_loss_tracker.result(),
            'kl_loss': self.kl_loss_tracker.result(),
            'classification_loss': self.classification_loss_tracker.result(),
            'contrastive_loss': self.contrastive_loss_tracker.result(),
        }
    @staticmethod
    def contrastive_loss(z1, z2, y, margin=0.0):
        d = tf.norm(z1 - z2, axis=2)
        loss = y * tf.square(d) + (1 - y) * tf.square(tf.maximum(margin - d, 0))
        return 1000 * tf.reduce_mean(loss)
    def calculate_loss(self, data):
        with tf.GradientTape() as tape:
            inputs = data[0]
            labels = data[1]
            z_mean, z_log_var, z = self.encoder(inputs)
            recon = self.decoder(z)
            logits = self.classifier(z)
            labels = tf.cast(labels, tf.float32)
            labels = tf.squeeze(labels, axis=-1)
            valid_mask = ~tf.math.is_nan(labels)
            filtered_labels = tf.boolean_mask(labels, valid_mask)
            filtered_logits = tf.boolean_mask(logits, valid_mask)
            classification_loss = tf.cond(
                tf.greater(tf.size(filtered_labels), 0),
                lambda: tf.reduce_mean(
                    keras.losses.sparse_categorical_crossentropy(filtered_labels, filtered_logits)
                ),
                lambda: tf.constant(0.0),
            )
            reconstruction_loss = 1000 * keras.losses.MeanSquaredError()(inputs, recon)
            kl = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 0.1 * tf.reduce_mean(tf.reduce_sum(kl, axis=1))
            contrastive_loss = tf.constant(0.0)
            total_loss = kl_loss + reconstruction_loss + classification_loss + contrastive_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return total_loss, tape
class SSVAE:
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.latent_dim = 2
        self.weights_path = None
        self.keras_model = self.build_keras_model()
    def prepare_data_for_keras_model(self, data):
        return self.__scale_and_transform_to_numpy(data)
    def __scale_and_transform_to_numpy(self, data):
        return np.where(data == 0, 0.0, 1.0)
    def build_keras_model(self):
        input_dim = self.input_dim
        # Encoder
        inputs = keras.Input(shape=input_dim)
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(128, activation='leaky_relu')(x)
        x = keras.layers.Dense(64, activation='leaky_relu')(x)
        x = keras.layers.Dense(32, activation='leaky_relu')(x)
        x = keras.layers.Dense(16, activation='leaky_relu')(x)
        z_mean = keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        # Classifier
        latent_in = keras.Input(shape=(self.latent_dim,))
        cx = keras.layers.Dense(16, activation='leaky_relu')(latent_in)
        cx = keras.layers.Dense(16, activation='leaky_relu')(cx)
        logits = keras.layers.Dense(10, activation='softmax')(cx)
        classifier = keras.Model(inputs=latent_in, outputs=logits, name='classifier')
        classifier.summary()
        # Decoder
        latent_in = keras.Input(shape=(self.latent_dim,))
        dx = keras.layers.Dense(16, activation='leaky_relu')(latent_in)
        dx = keras.layers.Dense(32, activation='leaky_relu')(dx)
        dx = keras.layers.Dense(64, activation='leaky_relu')(dx)
        dx = keras.layers.Dense(128, activation='leaky_relu')(dx)
        dx = keras.layers.Dense(input_dim[0] * input_dim[1])(dx)
        out = keras.layers.Reshape((input_dim[0], input_dim[1]))(dx)
        decoder = keras.Model(inputs=latent_in, outputs=out, name='decoder')
        decoder.summary()
        model = SemiSupervisedVariationalAutoEncoderKerasModel(encoder, decoder, classifier)
        model.build(input_shape=(None, *input_dim))
        model.compile(optimizer=keras.optimizers.Adam())
        return model
    def load_model_weights(self, weights_path):
        self.weights_path = weights_path
        self.keras_model.load_weights(self.weights_path)
    def predict(self, data):
        z, _, _ = self.keras_model.encoder(data)
        recon = self.keras_model.decoder(z).numpy()
        logits = self.keras_model.classifier(z).numpy()
        pred_class = np.argmax(logits, axis=1)
        pred_certainty = np.max(logits, axis=1)
        return z.numpy(), recon, pred_class, pred_certainty
    def fit(self, data, labels, weights_path):
        self.weights_path = weights_path
        labels = labels.reshape(-1, 1)
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        checkpoint = ModelCheckpoint(filepath=self.weights_path, monitor='loss', save_best_only=True, save_weights_only=True)
        plot_losses = PlotLosses()
        sample_weights = np.where(np.isnan(labels), 1.0, 1000.0)
        history = self.keras_model.fit(
            data,
            labels,
            epochs=1_000_000,
            batch_size=4 * 1024,
            sample_weight=sample_weights,
            validation_split=0.1,
            callbacks=[checkpoint, plot_losses],
        )
        return history
class SSCVAE(SSVAE):
    def build_keras_model(self):
        input_dim = (28, 28, 1)
        # Encoder
        inputs = keras.Input(shape=input_dim)
        x = keras.layers.Conv2D(5, (5, 5), activation='leaky_relu', padding='same')(inputs)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = keras.layers.Conv2D(5, (5, 5), activation='leaky_relu', padding='same')(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding='same')(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation='leaky_relu')(x)
        x = keras.layers.Dense(32, activation='leaky_relu')(x)
        x = keras.layers.Dense(16, activation='leaky_relu')(x)
        z_mean = keras.layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = keras.layers.Dense(self.latent_dim, name='z_log_var')(x)
        z = Sampling()([z_mean, z_log_var])
        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name='encoder')
        encoder.summary()
        # Classifier
        latent_in = keras.Input(shape=(self.latent_dim,))
        cx = keras.layers.Dense(16, activation='leaky_relu')(latent_in)
        cx = keras.layers.Dense(16, activation='leaky_relu')(cx)
        logits = keras.layers.Dense(10, activation='softmax')(cx)
        classifier = keras.Model(inputs=latent_in, outputs=logits, name='classifier')
        classifier.summary()
        # Decoder
        latent_in = keras.Input(shape=(self.latent_dim,))
        dx = keras.layers.Dense(16, activation='leaky_relu')(latent_in)
        dx = keras.layers.Dense(32, activation='leaky_relu')(dx)
        dx = keras.layers.Dense(128, activation='leaky_relu')(dx)
        dx = keras.layers.Dense(7 * 7 * 5, activation='leaky_relu')(dx)
        dx = keras.layers.Reshape((7, 7, 5))(dx)
        dx = keras.layers.Conv2DTranspose(5, (5, 5), strides=(2, 2), activation='leaky_relu', padding='same')(dx)
        out = keras.layers.Conv2DTranspose(input_dim[-1], (5, 5), strides=(2, 2), activation='sigmoid', padding='same')(dx)
        decoder = keras.Model(inputs=latent_in, outputs=out, name='decoder')
        decoder.summary()
        model = SemiSupervisedVariationalAutoEncoderKerasModel(encoder, decoder, classifier)
        model.build(input_shape=(None, *input_dim))
        model.compile(optimizer=keras.optimizers.Adam())
        return model