import datetime
import logging
import os
import time

import numpy as np
import tensorflow as tf
from sklearn.neighbors import KNeighborsClassifier
from wandb.keras import WandbMetricsLogger, WandbModelCheckpoint

import wandb
from metric.visualization import generate_and_plot_data, visualization

from .modules.discriminator import Discriminator
from .modules.embedder import Embedder
from .modules.generator import Generator
from .modules.model_utils import (
    MinMaxScaler,
    batch_generator,
    extract_time,
    random_generator,
    save_dict_to_json,
)
from .modules.recovery import Recovery
from .modules.supervisor import Supervisor

# logging.disable(logging.WARNING)


class TimeGAN(tf.keras.Model):
    def __init__(self, args):
        super().__init__()
        self.embedder = Embedder(args)
        self.recovery = Recovery(args)
        self.generator = Generator(args)
        self.supervisor = Supervisor(args)
        self.discriminator = Discriminator(args)
        self.mse = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        pass

    # E0_solver
    def recovery_forward(self, X, optimizer):
        # initial_hidden = self.embedder.initialize_hidden_state()
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = self.mse(X, X_tilde)
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss0, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss_T0

    # GS_solver
    def supervisor_forward(self, X, Z, optimizer):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            # H_hat = self.generator(Z, training=True)
            H_hat_supervise = self.supervisor(H, training=True)
            G_loss_S = self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])

        var_list = self.supervisor.trainable_weights
        grads = tape.gradient(G_loss_S, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return G_loss_S

    # G_solver
    def generator_forward(self, X, Z, optimizer=None, gamma=1):
        with tf.GradientTape() as tape:
            H = self.embedder(X)
            E_hat = self.generator(Z, training=True)

            # Supervisor & Recovery
            H_hat_supervise = self.supervisor(H, training=True)
            H_hat = self.supervisor(E_hat, training=True)
            X_hat = self.recovery(H_hat)

            # Discriminator
            Y_fake = self.discriminator(H_hat)
            Y_real = self.discriminator(H)
            Y_fake_e = self.discriminator(E_hat)

            # Generator loss
            G_loss_U = tf.math.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(Y_fake), Y_fake, from_logits=True
                )
            )
            G_loss_U_e = tf.math.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(Y_fake_e), Y_fake_e, from_logits=True
                )
            )
            G_loss_S = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(
                    H[:, 1:, :], H_hat_supervise[:, :-1, :]
                )
            )
            # Difference in "variance" between X_hat and X
            G_loss_V1 = tf.math.reduce_mean(
                tf.math.abs(
                    tf.math.sqrt(tf.nn.moments(X_hat, [0])[1] + 1e-6)
                    - tf.math.sqrt(tf.nn.moments(X, [0])[1] + 1e-6)
                )
            )
            # Difference in "mean" between X_hat and X
            G_loss_V2 = tf.math.reduce_mean(
                tf.math.abs((tf.nn.moments(X_hat, [0])[0]) - (tf.nn.moments(X, [0])[0]))
            )
            G_loss_V = G_loss_V1 + G_loss_V2
            # G_loss_V = tf.math.add(G_loss_V1, G_loss_V2)
            ## Sum of all G_losses
            G_loss = (
                G_loss_U
                + gamma * G_loss_U_e
                + 100 * tf.math.sqrt(G_loss_S)
                + 100 * G_loss_V
            )

        GS_var_list = (
            self.generator.trainable_weights + self.supervisor.trainable_weights
        )
        GS_grads = tape.gradient(G_loss, GS_var_list)
        optimizer.apply_gradients(zip(GS_grads, GS_var_list))

        return G_loss_U, G_loss_S, G_loss_V

    # D_solver
    def discriminator_forward(self, X, Z, optimizer=None, gamma=1, train_D=False):
        if train_D:
            with tf.GradientTape() as tape:
                H = self.embedder(X)
                E_hat = self.generator(Z)

                # Supervisor & Recovery
                H_hat_supervise = self.supervisor(H)
                H_hat = self.supervisor(E_hat)
                X_hat = self.recovery(H_hat)

                # Discriminator forward
                Y_fake = self.discriminator(H_hat)
                Y_real = self.discriminator(H)
                Y_fake_e = self.discriminator(E_hat)

                # Discriminator loss
                D_loss_real = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.ones_like(Y_real), Y_real, from_logits=True
                    )
                )
                D_loss_fake = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.zeros_like(Y_fake), Y_fake, from_logits=True
                    )
                )
                D_loss_fake_e = tf.math.reduce_mean(
                    tf.keras.losses.binary_crossentropy(
                        tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True
                    )
                )
                D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

            D_var_list = self.discriminator.trainable_weights
            D_grads = tape.gradient(D_loss, D_var_list)
            optimizer.apply_gradients(zip(D_grads, D_var_list))

        else:
            # Checking if D_loss > 0.15
            H = self.embedder(X)
            E_hat = self.generator(Z)

            # Supervisor & Recovery
            H_hat_supervise = self.supervisor(H)
            H_hat = self.supervisor(E_hat)
            X_hat = self.recovery(H_hat)

            # Discriminator forward
            Y_fake = self.discriminator(H_hat)
            Y_real = self.discriminator(H)
            Y_fake_e = self.discriminator(E_hat)

            # Discriminator loss
            D_loss_real = tf.math.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.ones_like(Y_real), Y_real, from_logits=True
                )
            )
            D_loss_fake = tf.math.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.zeros_like(Y_fake), Y_fake, from_logits=True
                )
            )
            D_loss_fake_e = tf.math.reduce_mean(
                tf.keras.losses.binary_crossentropy(
                    tf.zeros_like(Y_fake_e), Y_fake_e, from_logits=True
                )
            )
            D_loss = D_loss_real + D_loss_fake + gamma * D_loss_fake_e

        return D_loss

    # E_solver
    def embedding_forward_joint(self, X, optimizer, eta):
        with tf.GradientTape() as tape:
            H = self.embedder(X, training=True)
            X_tilde = self.recovery(H, training=True)
            E_loss_T0 = tf.math.reduce_mean(
                tf.keras.losses.mean_squared_error(X, X_tilde)
            )
            E_loss0 = 10 * tf.math.sqrt(E_loss_T0)

            H_hat_supervise = self.supervisor(H)
            G_loss_S = tf.math.reduce_mean(
                self.mse(H[:, 1:, :], H_hat_supervise[:, :-1, :])
            )
            E_loss = E_loss0 + eta * G_loss_S

        var_list = self.embedder.trainable_weights + self.recovery.trainable_weights
        grads = tape.gradient(E_loss, var_list)
        optimizer.apply_gradients(zip(grads, var_list))

        return E_loss

    # Inference
    def generate(self, Z, ori_data_num, ori_time, max_val, min_val, normalize=False):
        """
        Args:
            Z: input random noises
            ori_data_num: the first dimension of ori_data.shape
            ori_time: timesteps of original data
            max_val: the maximum value of MinMaxScaler(ori_data)
            min_val: the minimum value of MinMaxScaler(ori_data)
        Return:
            generated_data: synthetic time-series data
        """
        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        generated_data_curr = self.recovery(H_hat)
        generated_data_curr = generated_data_curr.numpy()
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, : ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        if normalize:
            generated_data = generated_data * max_val
            generated_data = generated_data + min_val

        return np.array(generated_data)

    # Inference by autoencoder
    def ae_generate(self, ori_data, ori_data_num, ori_time, max_val, min_val):
        H = self.embedder(ori_data)
        generated_data_curr = self.recovery(H)
        generated_data_curr = generated_data_curr.numpy()
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, : ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        return generated_data

    # Direct inference (testing)
    def generator_inference(self, z_dim, ori_data, model_dir):
        """This function is currently not used.
        Args:
            Z: input random noises
            ori_data: the original dataset (for information extraction)
            trained_model: trained Generator
        Return:
            generated_data: synthetic time-series data
        """
        no, seq_len, dim = np.asarray(ori_data).shape
        ori_time, max_seq_len = extract_time(ori_data)
        # Normalization
        _, min_val, max_val = MinMaxScaler(ori_data)

        if z_dim == -1:  # choose z_dim for the dimension of noises
            z_dim = dim
        Z = random_generator(no, z_dim, ori_time, max_seq_len)

        # Load models
        self.recovery.load_weights(model_dir)
        self.supervisor.load_weights(model_dir)
        self.generator.load_weights(model_dir)

        E_hat = self.generator(Z)
        H_hat = self.supervisor(E_hat)
        generated_data_curr = self.recovery(H_hat)
        generated_data = list()

        for i in range(ori_data_num):
            temp = generated_data_curr[i, : ori_time[i], :]
            generated_data.append(temp)

        # Renormalization
        generated_data = generated_data * max_val
        generated_data = generated_data + min_val

        return generated_data


def train_timegan(ori_data, dynamic_time, args):
    wandb.login()

    run = wandb.init(
        project="timegan" if not args["dp_training"] else "dptimegan", config=args
    )

    print("TRAINING WITH FOLLOWING CONFIG:")
    print(args)
    no, seq_len, dim = np.asarray(ori_data).shape

    if args["z_dim"] == -1:  # choose z_dim for the dimension of noises
        args["z_dim"] = dim

    ori_time, max_seq_len = extract_time(ori_data)
    # Normalization
    if args["normalization"]:
        ori_data, min_val, max_val = MinMaxScaler(ori_data)
        print("DATA is normalized")
    else:
        min_val = 0
        max_val = 1
        print("NO DATA normalized")

    model = TimeGAN(args)

    # Set up optimizers
    # if args['optimizer'] == 'adam':
    #     optimizer = tf.keras.optimizers.Adam(epsilon=args['epsilon'])

    optimizer = tf.keras.optimizers.legacy.Adam(epsilon=args["epsilon"])

    # print('Set up Tensorboard')
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = os.path.join("tensorboard", current_time + "-" + args["exp_name"])
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    # save args to tensorboard folder
    # save_dict_to_json(args['__dict__'], os.path.join(train_log_dir, 'params.json'))

    # 1. Embedding network training
    print("Start Embedding Network Training")
    start = time.time()
    for itt in range(args["embedding_iterations"]):
        X_mb, T_mb = batch_generator(
            ori_data, ori_time, args["batch_size"], use_tf_data=False
        )
        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        step_e_loss = model.recovery_forward(X_mb, optimizer)
        if itt % 100 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(args["embedding_iterations"])
                + ", e_loss: "
                + str(np.round(np.sqrt(step_e_loss), 4))
            )
            # Write to wandb
            wandb.log({"Embedding_loss": np.round(np.sqrt(step_e_loss), 4)}, step=itt)

    print("Finish Embedding Network Training")
    end = time.time()
    print("Train embedding time elapsed: {} sec".format(end - start))

    # 2. Training only with supervised loss
    print("Start Training with Supervised Loss Only")
    start = time.time()
    for itt in range(args["supervised_iterations"]):
        # for itt in range(1):
        X_mb, T_mb = batch_generator(ori_data, ori_time, args["batch_size"])
        Z_mb = random_generator(
            args["batch_size"], args["z_dim"], T_mb, args["max_seq_len"]
        )

        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        step_g_loss_s = model.supervisor_forward(X_mb, Z_mb, optimizer)
        if itt % 100 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(args["supervised_iterations"])
                + ", s_loss: "
                + str(np.round(np.sqrt(step_g_loss_s), 4))
            )
            # Write to wandb
            wandb.log(
                {"Supervised_loss": np.round(np.sqrt(step_g_loss_s), 4)}, step=itt
            )

    print("Finish Training with Supervised Loss Only")
    end = time.time()
    print("Train Supervisor time elapsed: {} sec".format(end - start))

    # 3. Joint Training
    print("Start Joint Training")
    start = time.time()
    for itt in range(args["joint_iterations"]):
        # Generator training (two times as discriminator training)
        for g_more in range(2):
            X_mb, T_mb = batch_generator(ori_data, ori_time, args["batch_size"])
            Z_mb = random_generator(
                args["batch_size"], args["z_dim"], T_mb, args["max_seq_len"]
            )

            X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

            step_g_loss_u, step_g_loss_s, step_g_loss_v = model.generator_forward(
                X_mb, Z_mb, optimizer
            )
            step_e_loss_t0 = model.embedding_forward_joint(X_mb, optimizer, args["eta"])

        # Discriminator training
        X_mb, T_mb = batch_generator(ori_data, ori_time, args["batch_size"])
        Z_mb = random_generator(
            args["batch_size"], args["z_dim"], T_mb, args["max_seq_len"]
        )

        X_mb = tf.convert_to_tensor(X_mb, dtype=tf.float32)
        Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)

        check_d_loss = model.discriminator_forward(X_mb, Z_mb, train_D=False)
        if check_d_loss > 0.15:
            step_d_loss = model.discriminator_forward(
                X_mb, Z_mb, optimizer, train_D=True
            )
        else:
            step_d_loss = check_d_loss

        if itt % 100 == 0:
            print(
                "step: "
                + str(itt)
                + "/"
                + str(args["joint_iterations"])
                + ", d_loss: "
                + str(np.round(step_d_loss, 4))
                + ", g_loss_u: "
                + str(np.round(step_g_loss_u, 4))
                + ", g_loss_s: "
                + str(np.round(np.sqrt(step_g_loss_s), 4))
                + ", g_loss_v: "
                + str(np.round(step_g_loss_v, 4))
                + ", e_loss_t0: "
                + str(np.round(np.sqrt(step_e_loss_t0), 4))
            )

            # Write to wandb
            wandb.log(
                {
                    "Joint/Discriminator": np.round(step_d_loss, 4),
                    "Joint/Generator": np.round(step_g_loss_u, 4),
                    "Joint/Supervisor": np.round(step_g_loss_s, 4),
                    "Joint/Moments": np.round(step_g_loss_v, 4),
                    "Joint/Embedding": np.round(step_e_loss_t0, 4),
                },
                step=itt,
            )

            Z_mb = random_generator(no, args["z_dim"], ori_time, args["max_seq_len"])
            Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
            syn_data = model.generate(Z_mb, no, ori_time, max_val, min_val)

            # if args["train_on_stress_data"]:
            #     fig = generate_and_plot_data(
            #         syn_stress=syn_data,
            #         ori_stress=ori_data,
            #         samples=no,
            #     )
            # else:
            #     fig = generate_and_plot_data(
            #         syn_no_stress=syn_data,
            #         ori_no_stress=ori_data,
            #         samples=no,
            #     )

            # wandb.log({"signals": fig})

            plot_pca_stress = visualization(ori_data[: len(syn_data)], syn_data, "pca")

            wandb.log({"pca_stress": wandb.Image(plot_pca_stress)})

            plot_tsne_stress = visualization(
                ori_data[: len(syn_data)], syn_data, "tsne"
            )

            wandb.log({"tsne_stress": wandb.Image(plot_tsne_stress)})

    print("Finish Joint Training")
    end = time.time()
    print("Train jointly time elapsed: {} sec".format(end - start))

    ## Synthetic data generation
    # Z_mb = random_generator(no, args['z_dim'], ori_time, args['max_seq_len'])
    # Z_mb = tf.convert_to_tensor(Z_mb, dtype=tf.float32)
    # generated_data = model.generate(Z_mb, no, ori_time, max_val, min_val)

    wandb.finish()

    return model, max_val, min_val, train_log_dir


# Build a Data set for Classifier Two sample Test
def build_classifier_dataset(
    synthMos,
    realMos,
    synthTestMos,
    realTestMos,
    realTestNoMos,
    seq_length,
    num_features,
):
    nn_train_mos = np.concatenate(
        [
            np.reshape(
                realMos, (realMos.shape[0], seq_length * num_features), order="F"
            ),
            np.reshape(
                synthMos, (synthMos.shape[0], seq_length * num_features), order="F"
            ),
        ]
    )
    nn_label_mos = np.concatenate(
        [np.zeros((realMos.shape[0])), np.ones((synthMos.shape[0]))]
    )

    nn_train_nomos = np.concatenate(
        [
            np.reshape(
                realNoMos, (realNoMos.shape[0], seq_length * num_features), order="F"
            ),
            np.reshape(
                synthNoMos, (synthNoMos.shape[0], seq_length * num_features), order="F"
            ),
        ]
    )
    nn_label_nomos = np.concatenate(
        [np.zeros((realNoMos.shape[0])), np.ones((synthNoMos.shape[0]))]
    )

    nn_test_mos = np.concatenate(
        [
            np.reshape(
                realTestMos,
                (realTestMos.shape[0], seq_length * num_features),
                order="F",
            ),
            np.reshape(
                synthTestMos,
                (synthTestMos.shape[0], seq_length * num_features),
                order="F",
            ),
        ]
    )
    nn_label_test_mos = np.concatenate(
        [np.zeros((realTestMos.shape[0])), np.ones((synthTestMos.shape[0]))]
    )

    nn_test_nomos = np.concatenate(
        [
            np.reshape(
                realTestNoMos,
                (realTestNoMos.shape[0], seq_length * num_features),
                order="F",
            ),
            np.reshape(
                synthTestNoMos,
                (synthTestNoMos.shape[0], seq_length * num_features),
                order="F",
            ),
        ]
    )
    nn_label_test_nomos = np.concatenate(
        [np.zeros((realTestNoMos.shape[0])), np.ones((synthTestNoMos.shape[0]))]
    )

    nn_train = np.concatenate([nn_train_mos, nn_train_nomos])
    nn_label = np.concatenate([nn_label_mos, nn_label_nomos])

    nn_test = np.concatenate([nn_test_mos, nn_test_nomos])
    nn_label_test = np.concatenate([nn_label_test_mos, nn_label_test_nomos])

    return nn_train, nn_label, nn_test, nn_label_test
