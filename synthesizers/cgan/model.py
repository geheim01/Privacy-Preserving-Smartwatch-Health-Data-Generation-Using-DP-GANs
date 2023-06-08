import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb
from metric.visualization import plot_signal_distributions, visualization
from sklearn.neighbors import KNeighborsClassifier
from synthesizers.utils.training import (
    buildDataCTST,
    generate_and_plot_data,
    make_subplots,
    synthetic_dataset,
)
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint


class ConditionalGAN(tf.keras.Model):
    def __init__(self, discriminator, generator, seq_length, latent_dim, num_features):
        super(ConditionalGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.seq_length = seq_length
        self.num_features = num_features
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
        self.disc_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    @property
    def metrics(self):
        return [self.gen_loss_tracker, self.disc_loss_tracker]

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(ConditionalGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        # Unpack the data
        real_seq, one_hot_labels = data

        batch_size = tf.shape(real_seq)[0]

        # generate latent space out of normal distribution
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), dtype=tf.dtypes.float64
        )

        # Decode the noise (guided by labels) to fake images.
        generated_seq = self.generator([random_latent_vectors, one_hot_labels])

        # Combine them with real images. Note that we are concatenating the labels
        # with these images here.
        fake_seq_and_real_seq = tf.concat([generated_seq, real_seq], axis=0)
        seq_one_hot_label_comb = tf.concat([one_hot_labels, one_hot_labels], axis=0)

        # Assemble labels discriminating real from fake images.
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )

        # Train the discriminator.
        with tf.GradientTape(persistent=True) as tape:
            predictions = self.discriminator(
                [fake_seq_and_real_seq, seq_one_hot_label_comb]
            )
            # var_list = self.discriminator.trainable_variables
            d_loss = self.loss_fn(labels, predictions)

            # grads = self.d_optimizer.compute_gradients(
            #     d_loss, var_list, gradient_tape=disc_tape
            # )

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)

        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Sample random points in the latent space.

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim), dtype=tf.dtypes.float64
        )

        # Assemble labels that say "all real images".
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that not to update the weights of the discriminator here)
        # Diversity term calculation like in https://arxiv.org/pdf/1901.09024.pdf
        G_z1 = self.generator(
            [
                random_latent_vectors[: (batch_size // 2)],
                one_hot_labels[: (batch_size // 2)],
            ]
        )

        G_z2 = self.generator(
            [
                random_latent_vectors[(batch_size // 2) :],
                one_hot_labels[(batch_size // 2) :],
            ]
        )

        # calculate Gradients for generator with diversity term
        with tf.GradientTape() as tape:
            fake_seq = self.generator([random_latent_vectors, one_hot_labels])

            g_diff = tf.reduce_mean(tf.abs(G_z1 - G_z2))

            z_diff = tf.reduce_mean(
                tf.abs(
                    random_latent_vectors[: (batch_size // 2)]
                    - [random_latent_vectors[(batch_size // 2) :]]
                )
            )

            # 8 is the importance of the diversity term
            L_z = (g_diff / z_diff) * 8

            predictions = self.discriminator([fake_seq, one_hot_labels])
            g_loss = self.loss_fn(misleading_labels, predictions) - tf.cast(
                L_z, dtype=tf.dtypes.float32
            )

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))

        # Monitor loss.
        self.gen_loss_tracker.update_state(g_loss)
        self.disc_loss_tracker.update_state(d_loss)
        return {
            "g_loss": self.gen_loss_tracker.result(),
            "d_loss": self.disc_loss_tracker.result(),
            "div_term": L_z,
        }

    ## LSTM Generator

    def conditional_generator(
        hidden_units,
        seq_length,
        latent_dim,
        num_features,
        activation_function,
        classes=2,
    ):
        # connect latent_space with a neuronal Net
        in_label = tf.keras.layers.Input(shape=(1,))
        # embedding for categorical input
        layer = tf.keras.layers.Embedding(classes, 2)(in_label)

        layer = tf.keras.layers.Dense(seq_length)(layer)

        # reshape to additional channel
        layer = tf.keras.layers.Reshape((seq_length, 1))(layer)

        # orginal sequence input
        in_seq = tf.keras.layers.Input(shape=(latent_dim,))

        # connect latent_space with a neuronal Net
        seq = tf.keras.layers.Dense(seq_length * num_features)(in_seq)

        # activation
        seq = tf.keras.layers.LeakyReLU()(seq)

        # reshape to additional channel
        seq = tf.keras.layers.Reshape((seq_length, num_features))(seq)

        # merge input with label
        merge = tf.keras.layers.Concatenate()([seq, layer])

        # LSTM layer block
        rnn = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(merge)

        rnn = tf.keras.layers.LSTM(hidden_units, return_sequences=True)(rnn)

        # output linear activation layer || RELU Actiovation Function - wandb best c2st score
        out_layer = tf.keras.layers.Dense(
            num_features, dtype="float64", activation=activation_function
        )(rnn)

        model = tf.keras.models.Model([in_seq, in_label], out_layer)

        return model

    ## LSTM Discriminator

    def conditional_discriminator(
        seq_length,
        num_features,
        filters,
        kernel_sizes=None,
        classes=2,
    ):
        # connect label input with a neuronal Net
        in_label = tf.keras.layers.Input(shape=(1,))
        # embedding for categorical input
        layer = tf.keras.layers.Embedding(classes, 2)(in_label)

        layer = tf.keras.layers.Dense(seq_length)(layer)

        layer = tf.keras.layers.Reshape((seq_length, 1))(layer)

        # input sequenz
        in_seq = tf.keras.layers.Input(shape=(seq_length, num_features))

        # merge label and sequenz
        merge = tf.keras.layers.Concatenate()([in_seq, layer])

        # time Series Classification from Scratch https://arxiv.org/abs/1611.06455
        # convolutional layer block
        # You can play here with different filters and kernel sizes
        conv1 = tf.keras.layers.Conv1D(
            # filters=32, kernel_size=8, strides=1, padding="same"
            filters=filters[0],
            # kernel_size=kernel_sizes[0],
            kernel_size=8,
            strides=1,
            padding="same",
        )(merge)
        conv1 = tf.keras.layers.BatchNormalization()(conv1)
        conv1 = tf.keras.layers.ReLU()(conv1)

        # convolutional layer block
        conv2 = tf.keras.layers.Conv1D(
            # filters=64, kernel_size=5, strides=1, padding="same"
            filters=filters[1],
            # kernel_size=kernel_sizes[1],
            kernel_size=5,
            strides=1,
            padding="same",
        )(conv1)
        conv2 = tf.keras.layers.BatchNormalization()(conv2)
        conv2 = tf.keras.layers.ReLU()(conv2)

        # convolutional layer block
        conv3 = tf.keras.layers.Conv1D(
            # filters=32, kernel_size=3, strides=1, padding="same"
            filters=filters[2],
            # kernel_size=kernel_sizes[2],
            kernel_size=3,
            strides=1,
            padding="same",
        )(conv2)
        conv3 = tf.keras.layers.BatchNormalization()(conv3)
        conv3 = tf.keras.layers.ReLU()(conv3)

        # average pooling
        gap = tf.keras.layers.GlobalAveragePooling1D()(conv3)

        # output activation
        out_layer = tf.keras.layers.Dense(1)(
            gap
        )  # omit sigmoid activation here to use Numerical stable Binary Crossentropy loss function

        model = tf.keras.models.Model([in_seq, in_label], out_layer)

        return model


# Custom GAN Monitor
class GANMonitor(tf.keras.callbacks.Callback):
    def __init__(
        self,
        trainmos,
        trainnomos,
        testmos,
        testnomos,
        randomTrainMos,
        randomTrainNoMos,
        randomTestMos,
        randomTestNoMos,
        num_features,
        # noise_multiplier,
        batch_size,
        save_path,
        dp=False,
        num_seq=1,
        seq_length=18,
    ):
        self.trainmos = trainmos
        self.trainnomos = trainnomos
        self.testmos = testmos
        self.testnomos = testnomos
        self.randomTrainMos = randomTrainMos
        self.randomTrainNoMos = randomTrainNoMos
        self.randomTestMos = randomTestMos
        self.randomTestNoMos = randomTestNoMos
        self.seq_length = seq_length
        self.save_path = save_path
        self.scorelist = []
        # how many sequences you want to plot in the grid
        self.num_seq = num_seq
        self.num_features = num_features
        # self.noise_multiplier = noise_multiplier
        self.batch_size = batch_size
        # self.dp = dp

    def on_epoch_end(self, epoch, logs=None):
        label_mos = tf.ones((self.randomTrainNoMos.shape[0], 1))
        label_nomos = tf.zeros((self.randomTrainNoMos.shape[0], 1))

        syn_train_mos = self.model.generator(
            [self.randomTrainMos, label_mos[: self.randomTrainMos.shape[0]]]
        )
        syn_train_no_mos = self.model.generator([self.randomTrainNoMos, label_nomos])
        syn_test_mos = self.model.generator(
            [self.randomTestMos, label_mos[: self.randomTestMos.shape[0]]]
        )
        syn_test_no_mos = self.model.generator(
            [self.randomTestNoMos, label_nomos[: self.randomTestNoMos.shape[0]]]
        )

        label_mos = tf.ones((self.randomTrainMos.shape[0], 1))
        label_nomos = tf.zeros((self.randomTrainMos.shape[0], 1))
        syn_stress = self.model.generator([self.randomTrainMos, label_mos])
        syn_no_stress = self.model.generator([self.randomTrainMos, label_nomos])

        nn_train, nn_label, nn_test, nn_label_test = buildDataCTST(
            syn_train_mos,
            syn_train_no_mos,
            self.trainmos,
            self.trainnomos,
            syn_test_mos,
            syn_test_no_mos,
            self.testmos,
            self.testnomos,
            seq_length=self.seq_length,
            num_features=self.num_features,
        )

        # Perform Classifier two sample test
        neigh = KNeighborsClassifier(2)
        neigh.fit(nn_train, nn_label)
        c2st_score = neigh.score(nn_test, nn_label_test)
        self.scorelist.append(c2st_score)

        wandb.log({"c2st_score": c2st_score})

        if c2st_score <= min(self.scorelist) and c2st_score < 0.85:
            print(c2st_score)
            self.model.generator.save(
                f"{self.save_path}generator_{epoch}e_{c2st_score}c2st_"
            )

        if (epoch) % 100 == 0:
            plot_pca_stress = visualization(
                self.trainmos[: len(syn_stress)], syn_stress, "pca"
            )

            plot_tsne_stress = visualization(
                self.trainmos[: len(syn_stress)], syn_stress, "tsne"
            )

            fig = generate_and_plot_data(
                syn_stress,
                syn_no_stress,
                self.trainmos,
                self.trainnomos,
                self.num_seq,
                self.seq_length,
            )
            # plot_pca_no_stress = visualization(
            #     self.trainnomos[: len(syn_no_stress)], syn_no_stress, "pca"
            # )
            # plot_tsne_no_stress = visualization(
            #     self.trainnomos[: len(syn_no_stress)], syn_no_stress, "tsne"
            # )

            wandb.log({"pca_stress": wandb.Image(plot_pca_stress)})
            wandb.log({"tsne_stress": wandb.Image(plot_tsne_stress)})
            wandb.log({"signals": fig})
