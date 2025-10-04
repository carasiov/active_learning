import pathlib
import os
data_path = pathlib.Path(os.getcwd())/'data'
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import keras
import tensorflow as tf
import tensorflow_probability as tfp
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
import pickle
import matplotlib.pyplot as plt
from keras.callbacks import Callback # type: ignore
from sklearn.utils import shuffle
import tensorflow.keras.backend as K # type: ignore
from tensorflow.keras.constraints import min_max_norm # type: ignore


class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.reconstruction_loss = []
        self.val_reconstruction_loss = []
        self.classification_loss = []
        self.val_classification_loss = []
        self.kl_loss = []
        self.val_kl_loss = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.reconstruction_loss.append(logs.get('reconstruction_loss'))
        self.val_reconstruction_loss.append(logs.get('val_reconstruction_loss'))
        self.classification_loss.append(logs.get('classification_loss'))
        self.val_classification_loss.append(logs.get('val_classification_loss'))
        self.kl_loss.append(logs.get('kl_loss'))
        self.val_kl_loss.append(logs.get('val_kl_loss'))

        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot Loss and Validation Loss separately
        axes[0].plot(self.losses, label='Training Loss')
        axes[0].plot(self.val_losses, label='Validation Loss')
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Loss and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot the other loss values
        ax2 = axes[1].twinx()  # Zweite y-Achse für KL Loss
        
        axes[1].plot(self.reconstruction_loss, label='Reconstruction Loss', color='blue')
        axes[1].plot(self.val_reconstruction_loss, label='Validation Reconstruction Loss', color='cyan')
        # axes[1].plot(self.classification_loss, label='Classification Loss', color='green')
        # axes[1].plot(self.val_classification_loss, label='Validation Classification Loss', color='lime')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Reconstruction and Classification Losses')
        axes[1].legend(loc='upper left')
        axes[1].grid(True)
        
        # KL Loss auf zweiter Achse plotten
        ax2.plot(self.kl_loss, label='KL Loss', color='red')
        ax2.plot(self.val_kl_loss, label='Validation KL Loss', color='orange')        
        ax2.legend(loc='upper right')

        ax3 = ax2.twinx()
        ax3.plot(self.classification_loss, label='Classification Loss', color='green')
        ax3.plot(self.val_classification_loss, label='Validation Classification Loss', color='lime')  
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig(f'models/progress/ssvae_loss_plot.png')
        plt.close()



class Sampling(keras.layers.Layer):
    """Sample from normal distribution N(mu,sigma**2) given implicitly by mu and ln(sigma**2).

    Note that
        N(mu,sigma**2) ~ mu + sigma*N(0,1) ~ mu + exp(0.5*log_var)*N(0,1).
    """
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.dist = tfp.distributions.Normal(loc=0., scale=1.)

    # def call(self, inputs):
    #     mu, log_var = inputs
    #     epsilon = self.dist.sample(tf.shape(mu))
    #     # In case of the normal distribution, we could as well use
    #     # epsilon = keras.backend.random_normal(shape=tf.shape(mu))
    #     return mu + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs, training=None):
        mu, log_var = inputs
        if True:#training:
            epsilon = self.dist.sample(tf.shape(mu))
            return mu + tf.exp(0.5 * log_var) * epsilon
        else:
            return mu


class VariationalAutoEncoderKerasModel(keras.Model):
    def __init__(self, encoder, decoder,  **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.encoder = encoder
        self.decoder = decoder        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")        
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    def get_config(self):
        """Speichert die Konfiguration des Modells."""
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Rekonstruiert das Modell aus der Konfiguration."""
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        return cls(encoder=encoder, decoder=decoder, **config)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z)

    @property
    def metrics(self):
        """Return list of metrics for this model

        Implementing this property ensures that reset_states() is called on each metric after each epoch. This way
        metrics like MAE are calculated per epoch, as we would expect. Without this property, we would have to call
        reset_states() ourselves, otherwise metric values would be averages since the start of the training.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        """Overrides behaviour when model.fit() is called

        For a guide on how to override the train_step method for custom training, see
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        """
        total_loss, tape = self.calculate_loss(data)        
        
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))        
        
        # Return a dict mapping metric names to current value
        # This will trigger updates for the progress bar and any callbacks
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result()            
        }

    def test_step(self, data):
        """Overrides behaviour when model.evaluate() is called

        For a guide on how to override the test_step method for custom training, see
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        """
        total_loss, tape = self.calculate_loss(data)
        
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result()
        }

    def calculate_loss(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction       = self.decoder(z)        
            
            reconstruction_loss  = keras.losses.MeanSquaredError()(data, reconstruction)

            kl_loss    = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss    = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            total_loss = self.gamma*reconstruction_loss + (1-self.gamma)*kl_loss

        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return total_loss, tape


class SemiSupervisedVariationalAutoEncoderKerasModel(keras.Model):
    def __init__(self, encoder, decoder, classifier,  **kwargs):
        super().__init__(**kwargs)
        self.gamma = tf.Variable(1.0, trainable=False, dtype=tf.float32)
        self.encoder = encoder
        self.decoder = decoder        
        self.classifier = classifier        
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")        
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
        self.classification_loss_tracker = keras.metrics.Mean(name="classification_loss")
        self.contrastive_loss_tracker = keras.metrics.Mean(name="contrastive_loss")

    def get_config(self):
        """Speichert die Konfiguration des Modells."""
        config = super().get_config()
        config.update({
            "encoder": keras.saving.serialize_keras_object(self.encoder),
            "decoder": keras.saving.serialize_keras_object(self.decoder),
            "classifier": keras.saving.serialize_keras_object(self.classifier)
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Rekonstruiert das Modell aus der Konfiguration."""
        encoder = keras.saving.deserialize_keras_object(config.pop("encoder"))
        decoder = keras.saving.deserialize_keras_object(config.pop("decoder"))
        classifier = keras.saving.deserialize_keras_object(config.pop("classifier"))
        return cls(encoder=encoder, decoder=decoder, classifier=classifier,**config)

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        return self.decoder(z_mean), self.classifier(z_mean)

    @property
    def metrics(self):
        """Return list of metrics for this model

        Implementing this property ensures that reset_states() is called on each metric after each epoch. This way
        metrics like MAE are calculated per epoch, as we would expect. Without this property, we would have to call
        reset_states() ourselves, otherwise metric values would be averages since the start of the training.
        """
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
            self.classification_loss_tracker,
            self.contrastive_loss_tracker,
        ]

    def train_step(self, data):
        """Overrides behaviour when model.fit() is called

        For a guide on how to override the train_step method for custom training, see
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        """
        total_loss, tape = self.calculate_loss(data)
                
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)

        # Update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))        
        
        # Return a dict mapping metric names to current value
        # This will trigger updates for the progress bar and any callbacks
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "contrastive_loss":    self.contrastive_loss_tracker.result()
        }

    def test_step(self, data):
        """Overrides behaviour when model.evaluate() is called

        For a guide on how to override the test_step method for custom training, see
        https://www.tensorflow.org/guide/keras/customizing_what_happens_in_fit
        """
        self.calculate_loss(data)
                       
        return {
            "loss":                self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss":             self.kl_loss_tracker.result(),
            "classification_loss": self.classification_loss_tracker.result(),
            "contrastive_loss":    self.contrastive_loss_tracker.result()
        }
    
    @staticmethod
    def contrastive_loss(z1, z2, y, margin=0.0):
        d = tf.norm(z1 - z2, axis=2)  # Euclidean distance
        loss = y * tf.square(d)  + (1 - y) * tf.square(tf.maximum(margin - d, 0))
        return 1000 * tf.reduce_mean(loss)
    
    def calculate_loss(self, data):
        with tf.GradientTape() as tape:
            input = data[0]
            labels = data[1]  # Labels für Classification Loss            

            # **Encode input**
            z_mean, z_log_var, z = self.encoder(input)
            mu_recon = self.decoder(z)           
            classification_pred = self.classifier(z)

          
            contrastive_loss = 0.0


            # Classifier loss
            labels = tf.cast(labels, tf.float32)
            labels = tf.squeeze(labels, axis=-1)

            valid_mask = ~tf.math.is_nan(labels)
            filtered_labels = tf.boolean_mask(labels, valid_mask)
            filtered_predictions = tf.boolean_mask(classification_pred, valid_mask)
            
            classification_loss = tf.cond(
                tf.greater(tf.size(filtered_labels), 0),
                lambda: tf.reduce_mean(keras.losses.sparse_categorical_crossentropy(filtered_labels, filtered_predictions)),
                lambda: tf.constant(0.0)
            )     
            #classification_loss = 0.0

           
            # **Reconstruction Loss**
            reconstruction_loss = 1000 * keras.losses.MeanSquaredError()(input, mu_recon)

            # **KL-Divergenz Loss**
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = 0.1 * tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))

            # **Gesamter Loss**
            total_loss = (kl_loss 
                          + contrastive_loss
                          + reconstruction_loss + classification_loss)

        # **Update der Metriken**
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        self.classification_loss_tracker.update_state(classification_loss)
        self.contrastive_loss_tracker.update_state(contrastive_loss)
        return total_loss, tape



class GammaScheduler(keras.callbacks.Callback):
    def __init__(self, lambda_decay):
        super().__init__()
        self.lambda_decay = lambda_decay  # Steuert, wie schnell gamma fällt

    def on_epoch_begin(self, epoch, logs=None):
        new_gamma = 0.98 + 0.02 * np.exp(-self.lambda_decay * epoch)
        new_gamma = 0.98
        # Aktualisiere gamma im Modell
        self.model.gamma.assign(new_gamma)
        print(f"Epoch {epoch}: Set gamma to {new_gamma:.4f}")


class GammaSchedulerAB(keras.callbacks.Callback):
    def __init__(self, epoch_A, epoch_B):
        super().__init__()
        self.epoch_A = epoch_A  # Epoche, bis zu der gamma = 1 bleibt
        self.epoch_B = epoch_B  # Epoche, bei der gamma 0 erreicht
        self.reconstruction_loss = None
        self.start_transition = False

    def on_epoch_end(self, epoch, logs={}):        
        self.reconstruction_loss = logs.get('reconstruction_loss')
        if (self.reconstruction_loss < 0.35) & (not self.start_transition):
            self.epoch_A = epoch            

    def on_epoch_begin(self, epoch, logs=None):        
        if epoch < self.epoch_A:
            new_gamma = 1.0
        elif self.epoch_A <= epoch <= (self.epoch_A + self.epoch_B):
            self.start_transition = True
            new_gamma = 1.0 - (epoch - self.epoch_A) / self.epoch_B
        else:
            new_gamma = 0.0
        
        new_gamma = 0.0
        # Aktualisiere gamma im Modell
        self.model.gamma.assign(new_gamma)
        print(f"Epoch {epoch}: Set gamma to {new_gamma:.4f}")



class VAE:

    """ Model class"""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.latent_dim = 8
        self.weights_path = None
        self.keras_model = self.build_keras_model()
        self.scalers = None
        self.normalize = False
        self.cols = None
        self.medians = None

    def prepare_data_for_keras_model(self, df):
        if self.normalize:
            df, self.cols, self.medians = self.__normalize(df, variables_to_normalize=['helium_level'])        
        scaled_data_np = self.__scale_and_transform_to_numpy(df)
        return scaled_data_np

    @staticmethod
    def __normalize(df, variables_to_normalize=None):
        """ This function subtracts the median from every single timerow. This is done for each variable separatly """
        df = df.copy()

        cols = {}
        for i, var in enumerate(variables_to_normalize):
            cols[var] = [x for x in df.columns if x[0] == var]

        # Median für die entsprechenden Spalten (je Zeile) berechnen
        medians = {}
        for i, var in enumerate(cols):  # df.columns.get_level_values(0).drop_duplicates()):
            medians[var] = df[cols[var]].median(axis=1)

        # Median von den entsprechenden Spaltenwerten abziehen
        for i, var in enumerate(cols):  # df.columns.get_level_values(0).drop_duplicates()):
            df[cols[var]] = df[cols[var]].subtract(medians[var], axis=0)

        return df, cols, medians

   
    def __scale_and_transform_to_numpy(self, df):
        """
        Transformiert den DataFrame in ein Tensor-geeignetes Format und normalisiert die Daten,
        wobei der StandardScaler so verwendet wird, dass die Skalierung für alle Zeitschritte
        einer Variablen konsistent bleibt.

        :param df: Pandas DataFrame mit:
                   - Doppelindex (Strahler- und System-Seriennummern als Index).
                   - Doppelindex für Spalten (Variablen und Timesteps).
        :return: NumPy-Array der Form (n_samples, n_variables, n_timesteps).
        """
        # Überprüfen, ob der Spaltenindex ein MultiIndex ist
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("Die Spalten des DataFrames müssen einen MultiIndex haben (variable, timestep).")

        # MultiIndex trennen: Variablen und Timesteps extrahieren
        variables = df.columns.get_level_values(0).unique()

        # Normalisierung der Daten
        normalized_data = []

        if self.scalers is None:
            print("No fitted scalers available. Refitting...")
            fit_scalers = True
            self.scalers = {}
        else:
            print('Using given fitted scalers')
            fit_scalers = False

        for variable in variables:
            # Alle Spalten einer Variable auswählen (alle Zeitschritte)
            variable_data = df.xs(variable, axis=1, level=0).values  # Shape: (n_samples, timesteps)

            # Zeitschritte untereinander packen (flatten)
            flattened_data = variable_data.flatten().reshape(-1, 1)  # Shape: (n_samples * n_timesteps, 1)

            # StandardScaler fitten
            if fit_scalers:
                self.scalers[variable] = StandardScaler()
                self.scalers[variable].fit(flattened_data)

            # Den gleichen Skaler auf jeden Zeitschritt anwenden
            variable_data_scaled = self.scalers[variable].transform(variable_data.reshape(-1, 1))  # Shape bleibt gleich
            variable_data_scaled = variable_data_scaled.reshape(variable_data.shape)  # Zurück zur ursprünglichen Form

            # Hinzufügen der normalisierten Daten zur Liste
            normalized_data.append(variable_data_scaled)

        # Stapeln der normalisierten Daten entlang der Variablen-Achse
        normalized_data = np.stack(normalized_data, axis=2)  # Shape: (n_samples, n_timesteps, n_variables)

        return normalized_data

    def build_keras_model(self):
        input_dim = self.input_dim
        
        # ENCODER
        inputs = keras.Input(shape=input_dim)
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(128, activation="tanh")(x)
        x = keras.layers.Dense(64, activation="tanh")(x)
        x = keras.layers.Dense(32, activation="tanh")(x)
        x = keras.layers.Dense(16, activation="tanh")(x)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x) 
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # DECODER
        latent = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(16, activation="tanh")(latent)
        x = keras.layers.Dense(32, activation="tanh")(x)
        x = keras.layers.Dense(64, activation="tanh")(x)
        x = keras.layers.Dense(128, activation="tanh")(x)
        x = keras.layers.Dense(input_dim[0] * input_dim[1])(x)
        outputs = keras.layers.Reshape((input_dim[0], input_dim[1]))(x)

        decoder = keras.Model(inputs=latent, outputs=outputs, name="decoder")
        decoder.summary()
        
        model = VariationalAutoEncoderKerasModel(encoder, decoder)

        # Modell mit spezifischem Input-Shape bauen
        model.build(input_shape=(None, *input_dim))  # None für die Batch-Größe

        model.compile(optimizer=keras.optimizers.Adam())

        return model

    def load_model_weights(self, weights_path):
        self.weights_path = weights_path
        self.keras_model.load_weights(self.weights_path)

    def predict(self, df):
        with open("models/scalers.pkl", "rb") as f:
            self.scalers = pickle.load(f)
        scaled_data_np = self.prepare_data_for_keras_model(df=df)
        output_encoder, _, _ = self.keras_model.encoder(scaled_data_np)
        latent_df = pd.DataFrame(data=output_encoder.numpy(), index=df.index, columns=['latent_'+str(i) for i in range(self.latent_dim)])

        output_decoder = self.keras_model.decoder(output_encoder).numpy()

        # Rescale the reconstructed data
        list_of_rescaled_variables = [self.scalers[variable].inverse_transform(
            output_decoder[:,:,i].reshape(-1,1)).reshape(-1,self.input_dim[0]) for i, variable in enumerate(self.scalers)]

        # Transform the arrays from list_of_rescaled_variables to a DataFrame
        rescaled_data = np.concatenate(list_of_rescaled_variables, axis=1)
        columns = pd.MultiIndex.from_tuples([(variable, str(i)) for variable in self.scalers.keys() for i in range(self.input_dim[0])])
        rescaled_df = pd.DataFrame(data=rescaled_data, index=df.index, columns=columns)

        # Add the median back to the data
        if self.normalize:
            for variable in self.cols:
                for col in self.cols[variable]:
                    rescaled_df[col] = rescaled_df[col] + self.medians[variable]

        return latent_df, rescaled_df

    def fit(self, df, weights_path):
        self.weights_path = weights_path


        # Fit scalers and scale data and
        self.scalers = None
        scaled_data_np = self.prepare_data_for_keras_model(df=df)
        with open("models/scalers.pkl", "wb") as f:
            pickle.dump(self.scalers, f)

      
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',  
            patience=20,  
            restore_best_weights=True 
        )

        checkpoint = ModelCheckpoint(
            filepath=self.weights_path,  
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True
        )

        plot_losses = PlotLosses()

        gamma_scheduler = GammaScheduler(lambda_decay=0.1)
              
        history = self.keras_model.fit(
            scaled_data_np,
            epochs=1000000,  
            batch_size=16384,
            validation_split=0.1,           
            callbacks=[early_stopping, 
                       checkpoint, 
                       plot_losses, 
                       gamma_scheduler]
        )

        return history


class CVAE(VAE):

    def build_keras_model(self):
        input_dim = self.input_dim

        """ Why do i use tanh as activation function?
        Because using relu the reconstruction loss is not decreasing. Maybe because of dead neurons?"""
        
        # ENCODER
        inputs = keras.Input(shape=input_dim)        
        x = keras.layers.Conv1D(8, 3, activation="tanh", padding="same")(inputs)
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.Conv1D(8, 3, activation="tanh", padding="same")(x)
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.Conv1D(8, 3, activation="tanh", padding="same")(x)        
        x = keras.layers.MaxPooling1D(2, padding="same")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(16, activation="tanh")(x)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x) 
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

      
        # DECODER
        latent = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(40, activation="tanh")(latent)
        x = keras.layers.Reshape((20, 2))(x)
        x = keras.layers.Conv1DTranspose(8, 3, activation="tanh", padding="same")(x)
        x = keras.layers.UpSampling1D(2)(x)
        x = keras.layers.Conv1DTranspose(8, 3, activation="tanh", padding="same")(x)
        x = keras.layers.UpSampling1D(2)(x)        
        x = keras.layers.Conv1DTranspose(8, 3, activation="tanh", padding="same")(x)
        x = keras.layers.UpSampling1D(2)(x)
        outputs = keras.layers.Conv1DTranspose(input_dim[-1], 3, padding="same")(x)
        

        decoder = keras.Model(inputs=latent, outputs=outputs, name="decoder")
        decoder.summary()
        
        model = VariationalAutoEncoderKerasModel(encoder, decoder)

        # Modell mit spezifischem Input-Shape bauen
        model.build(input_shape=(None, *input_dim))  # None für die Batch-Größe

        model.compile(optimizer=keras.optimizers.Adam())

        return model


class SSVAE:

    """ Model class"""
    def __init__(self, input_dim):
        self.input_dim = input_dim
        self.latent_dim = 2

        self.weights_path = None
        self.keras_model = self.build_keras_model()
        self.scalers = None
        self.normalize = True
        self.cols = None
        self.medians = None

    def prepare_data_for_keras_model(self, data):        
       
        scaled_data_np = self.__scale_and_transform_to_numpy(data)
        return scaled_data_np

      
    def __scale_and_transform_to_numpy(self, data):

        normalized_data = np.where(data == 0, 0.0, 1.0)    

        return normalized_data

    def build_keras_model(self):
        input_dim = self.input_dim
        
        # ENCODER
        inputs = keras.Input(shape=input_dim)
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(128, activation="leaky_relu")(x)
        x = keras.layers.Dense(64, activation="leaky_relu")(x)
        x = keras.layers.Dense(32, activation="leaky_relu")(x)
        x = keras.layers.Dense(16, activation="leaky_relu")(x)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x) 
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Classifier
        latent = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(16, activation="leaky_relu")(latent)
        x = keras.layers.Dense(16, activation="leaky_relu")(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)
        classifier = keras.Model(inputs=latent, outputs=outputs, name="classifier")
        classifier.summary()
      
        # DECODER MU
        latent = keras.Input(shape=(self.latent_dim,))        
        x = keras.layers.Dense(16, activation="leaky_relu")(latent)
        x = keras.layers.Dense(32, activation="leaky_relu")(x)
        x = keras.layers.Dense(64, activation="leaky_relu")(x)
        x = keras.layers.Dense(128, activation="leaky_relu")(x)
        x = keras.layers.Dense(input_dim[0] * input_dim[1])(x)
        output = keras.layers.Reshape((input_dim[0], input_dim[1]))(x)        
        
        decoder = keras.Model(inputs=latent, outputs=output, name="decoder")
        decoder.summary()
        
        model = SemiSupervisedVariationalAutoEncoderKerasModel(encoder, decoder, classifier)

        # Modell mit spezifischem Input-Shape bauen
        model.build(input_shape=(None, *input_dim))  # None für die Batch-Größe

        model.compile(optimizer=keras.optimizers.Adam())

        return model

    def load_model_weights(self, weights_path):
        self.weights_path = weights_path
        self.keras_model.load_weights(self.weights_path)

    def predict(self, data):
        
        #normalized_data = self.prepare_data_for_keras_model(data=data)
     
        latent, _, _ = self.keras_model.encoder(data)
       
        #output_decoder = self.keras_model.decoder(output_encoder).numpy()
        recon = self.keras_model.decoder(latent).numpy()       

        output_classifier = self.keras_model.classifier(latent).numpy()
        pred_class = np.argmax(output_classifier, axis=1)
        output_certainty = np.max(output_classifier, axis=1)

        return latent.numpy(), recon, pred_class, output_certainty

    def fit(self, data, labels, weights_path):
        self.weights_path = weights_path

        labels = labels.reshape(-1, 1)
        # # One-hot encode the labels, preserving NaNs
        # num_classes = 10  # Adjust this based on the number of classes
        # one_hot_labels = np.zeros((labels.shape[0], num_classes))
        # for i, label in enumerate(labels):
        #     if np.isnan(label):
        #         one_hot_labels[i, :] = np.nan  # Preserve NaN
        #     else:
        #         one_hot_labels[i, int(label)] = 1
        # labels = one_hot_labels        
                      
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',  
            patience=20,  
            restore_best_weights=True 
        )

        checkpoint = ModelCheckpoint(
            filepath=self.weights_path,  
            monitor='loss',
            save_best_only=True,
            save_weights_only=True
        )

        plot_losses = PlotLosses()

        #gamma_scheduler = GammaSchedulerAB(epoch_A=100, epoch_B=200)

        # Sample-Gewichte: Höhere Gewichtung für vorhandene Labels
        sample_weights = np.where(np.isnan(labels), 1.0, 1000.0)        
    
        #normalized_data = self.prepare_data_for_keras_model(data=data)

        # Modelltraining mit expliziter Validierungsmenge
        history = self.keras_model.fit(
                            data, labels,
                            epochs=1000000,
                            batch_size=4*1024,
                            sample_weight=sample_weights,
                            validation_split=0.1,                            
                            callbacks=[checkpoint, plot_losses]#, gamma_scheduler]
                            )     

        return history


class SSCVAE(SSVAE):
    def build_keras_model(self):
        input_dim = (28, 28, 1)  # self.input_dim
        
         # ENCODER
        inputs = keras.Input(shape=input_dim)        
        x = keras.layers.Conv2D(5, (5, 5), activation="leaky_relu", padding="same")(inputs)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)        
        x = keras.layers.Conv2D(5, (5, 5), activation="leaky_relu", padding="same")(x)
        x = keras.layers.AveragePooling2D(pool_size=(2, 2), padding="same")(x)
        x = keras.layers.Flatten()(x)
        x = keras.layers.Dense(128, activation="leaky_relu")(x)
        x = keras.layers.Dense(32, activation="leaky_relu")(x)
        x = keras.layers.Dense(16, activation="leaky_relu")(x)

        z_mean = keras.layers.Dense(self.latent_dim, name="z_mean")(x) 
        z_log_var = keras.layers.Dense(self.latent_dim, name="z_log_var")(x)
        z = Sampling()([z_mean, z_log_var])

        encoder = keras.Model(inputs=inputs, outputs=[z_mean, z_log_var, z], name="encoder")
        encoder.summary()

        # Classifier
        latent = keras.Input(shape=(self.latent_dim,))
        x = keras.layers.Dense(16, activation="leaky_relu")(latent)
        x = keras.layers.Dense(16, activation="leaky_relu")(x)
        outputs = keras.layers.Dense(10, activation="softmax")(x)
        classifier = keras.Model(inputs=latent, outputs=outputs, name="classifier")
        classifier.summary()
        
        
        # DECODER
        latent = keras.Input(shape=(self.latent_dim,))        
        x = keras.layers.Dense(16, activation="leaky_relu")(latent)
        x = keras.layers.Dense(32, activation="leaky_relu")(x)
        x = keras.layers.Dense(128, activation="leaky_relu")(x)
         
        x = keras.layers.Dense(7 * 7 * 5, activation="leaky_relu")(x)
        x = keras.layers.Reshape((7, 7, 5))(x)
        x = keras.layers.Conv2DTranspose(5, (5, 5), strides=(2, 2), activation="leaky_relu", padding="same")(x)
        output = keras.layers.Conv2DTranspose(input_dim[-1], (5, 5), strides=(2, 2), activation="sigmoid", padding="same")(x)

        decoder = keras.Model(inputs=latent, outputs=output, name="decoder")
        decoder.summary()
        
        model = SemiSupervisedVariationalAutoEncoderKerasModel(encoder, decoder, classifier)

        # Modell mit spezifischem Input-Shape bauen
        model.build(input_shape=(None, *input_dim))  # None für die Batch-Größe

        model.compile(optimizer=keras.optimizers.Adam())

        return model
