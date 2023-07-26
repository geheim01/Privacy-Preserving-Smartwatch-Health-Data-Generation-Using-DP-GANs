# from stress_detector.data.datatype import DataType
from random import shuffle

import numpy as np
import tensorflow as tf
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

from stress_detector import constants
from stress_detector.data.datatype import DataType


def build_model(
    num_signals: int = 6, num_output_class: int = 2
) -> tf.keras.models.Sequential:
    # Define the model architecture
    model = tf.keras.Sequential()
    # input_shape = 14 Signale (bei uns max. 6) X 210 Inputs (aus Tabelle nach Fourier)
    model.add(tf.keras.layers.InputLayer(input_shape=[num_signals, 210, 1]))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"
        )
    )
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(
        tf.keras.layers.Conv2D(
            filters=64, activation="relu", kernel_size=(1, 3), strides=1, padding="same"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(1, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(
        tf.keras.layers.Dense(
            units=128, activation="relu", kernel_initializer="glorot_uniform"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(
        tf.keras.layers.Dense(
            units=64, activation="relu", kernel_initializer="glorot_uniform"
        )
    )
    model.add(tf.keras.layers.Dropout(rate=0.3))
    # Anzahl der Units = Anzahl der Klassen (2 - non-stress vs stress)
    model.add(
        tf.keras.layers.Dense(units=num_output_class, activation="sigmoid")
    )  # sigmoid statt softmax, da nur 2 Klassen

    optimizer = "rmsprop"

    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
    )
    return model


def train(
    signals,
    all_subjects_X,
    all_subjects_y,
    data_type: DataType,
    num_epochs: int,
    with_loso: bool,
    num_syn_samples: int = 1,
    additional_name: str = None,
    test_sub: int = None,
):
    print(f"DataType: {data_type}")
    print(f'Signals: {" ".join(signals)}')
    print(f"Number of signals: {len(signals)}")
    # global y_train

    model_path = ""
    groups_set = []
    # changed - no filter for smartwatch
    # all_subjects_X_os = filter_for_smartwatch_os(os, all_subjects_X)
    all_subjects_X_os = all_subjects_X
    if data_type in (
        DataType.DGAN,
        DataType.CGAN_LSTM,
        DataType.CGAN_FCN,
        DataType.CGAN_TRANSFORMER,
        DataType.TIMEGAN,
        DataType.DPCGAN,
    ):
        groups_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        subject_ids = constants.SUBJECT_IDS + [
            f"SYN_{num_syn_samples}"
        ]  # ids for subjects in WESAD dataset
    if data_type == DataType.REAL:
        groups_set = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        subject_ids = constants.SUBJECT_IDS  # ids for subjects in WESAD dataset
    num_signals = len(
        signals
    )  # Number of signals in the WESAD dataset measured by the empatica e4
    num_output_class = 2  # Number of output classes (2 - non-stress vs stress)
    # num_epochs = 100

    if with_loso:
        base_path = "models/stress_detector/loso"
    else:
        base_path = "models/stress_detector/tstr"  # train syn test real

    if test_sub is not None:
        test_sub_index = constants.SUBJECT_IDS.index(test_sub)
    print(f"BASE PATH: {base_path}")
    if with_loso:
        for i in groups_set:
            test_index = groups_set[i]
            train_index = [x for x in groups_set if x != test_index]

            print(test_index)
            if test_sub is not None:
                if test_sub_index != test_index:
                    print("SKIP training for test_index", test_index)
                    continue
            print(f"Train on: {train_index}")
            print(f"Test  on: {test_index}")

            X_train = np.concatenate(
                np.array([all_subjects_X_os[x] for x in train_index], dtype=object)
            )
            y_train = np.concatenate(
                np.array([all_subjects_y[y] for y in train_index], dtype=object)
            )

            X_test = all_subjects_X_os[test_index]
            y_test = all_subjects_y[test_index]

            weight_balance = y_train.tolist().count(0) / y_train.tolist().count(1)

            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            y_train = tf.keras.utils.to_categorical(y_train, num_output_class)
            y_test = tf.keras.utils.to_categorical(y_test, num_output_class)

            tf.keras.backend.clear_session()

            model = build_model(num_signals, num_output_class)

            print(f"data_type == DataType.DGAN: {data_type == DataType.DGAN}")
            print(f"data_type == DataType.CGAN: {data_type == DataType.CGAN_LSTM}")
            print(f"data_type == DataType.DPCGAN: {data_type == DataType.DPCGAN}")
            print(f"data_type == DataType.TIMEGAN: {data_type == DataType.TIMEGAN}")
            print(f"data_type == DataType.REAL: {data_type == DataType.REAL}")

            if data_type == DataType.REAL:
                model_path = f"{base_path}/real/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"  # Path to save the model file
            if data_type == DataType.DGAN:
                model_path = f"{base_path}/syn/dgan_30000/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"
            if data_type == DataType.TIMEGAN:
                model_path = f"{base_path}/syn/timegan/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"
            if data_type == DataType.CGAN_LSTM:
                model_path = f"{base_path}/syn/cgan/no_dp/lstm/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"
            if data_type == DataType.CGAN_FCN:
                model_path = f"{base_path}/syn/cgan/no_dp/fcn/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"
            if data_type == DataType.CGAN_TRANSFORMER:
                model_path = f"{base_path}/syn/cgan/no_dp/transformer/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"
            if data_type == DataType.DPCGAN:
                model_path = f"{base_path}/syn/cgan/dp/{num_epochs}_epochs/{num_syn_samples}_subjects/wesad_s{subject_ids[test_index]}.h5"

            print(f"MODEL PATH: {model_path}")

            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    model_path,
                    # f"models/{smart_os}/wesad_{smart_os}_binary_s_syn_{num_epochs}.h5",  # Path to save the model file
                    # f"models/syn/wesad_{smart_os}_binary_s_syn_{num_epochs}.h5",  # Path to save the model file
                    monitor="loss",  # The metric name to monitor
                    save_best_only=True,  # If True, it only saves the "best" model according to the quantity monitored
                ),
                tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10),
            ]

            history = model.fit(
                X_train,
                y_train,
                epochs=num_epochs,
                batch_size=50,
                verbose=1,
                class_weight={
                    0: 1,
                    1: weight_balance,
                },  # to address the imbalance of the class labels
                callbacks=callbacks,
            )

    else:
        X = all_subjects_X_os[-1]
        y = all_subjects_y[-1]

        ind_list = [i for i in range(len(X))]
        shuffle(ind_list)
        X = X[ind_list, :, :]
        y = y[ind_list,]

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=1
        )

        print(y_train[:10])

        # weight_balance = y_train.tolist().count(0) / y_train.tolist().count(1)

        if y_train.tolist().count(1) != 0:
            weight_balance = y_train.tolist().count(0) / y_train.tolist().count(1)
        else:
            weight_balance = None

        X_train = np.asarray(X_train)
        X_test = np.asarray(X_test)
        y_train = np.asarray(y_train)
        y_test = np.asarray(y_test)

        y_train = tf.keras.utils.to_categorical(y_train, num_output_class)
        y_test = tf.keras.utils.to_categorical(y_test, num_output_class)

        tf.keras.backend.clear_session()

        model = build_model(num_signals, num_output_class)

        print(f"data_type == DataType.DGAN: {data_type == DataType.DGAN}")
        print(f"data_type == DataType.CGAN_LSTM: {data_type == DataType.CGAN_LSTM}")
        print(f"data_type == DataType.DPCGAN: {data_type == DataType.DPCGAN}")
        print(f"data_type == DataType.TIMEGAN: {data_type == DataType.TIMEGAN}")

        if data_type == DataType.DGAN:
            model_path = f"{base_path}/syn/dgan_30000/{num_epochs}/wesad.h5"
        if data_type == DataType.CGAN_LSTM:
            model_path = f"{base_path}/syn/cgan/no_dp/lstm/{num_epochs}/wesad.h5"
        if data_type == DataType.CGAN_FCN:
            model_path = f"{base_path}/syn/cgan/no_dp/fcn/{num_epochs}/wesad.h5"
        if data_type == DataType.CGAN_TRANSFORMER:
            model_path = f"{base_path}/syn/cgan/no_dp/transformer/{num_epochs}/wesad.h5"
        if data_type == DataType.DPCGAN:
            model_path = (
                f"{base_path}/syn/cgan/dp/{num_epochs}/{additional_name}_wesad.h5"
            )
        if data_type == DataType.TIMEGAN:
            model_path = f"{base_path}/syn/timegan/{num_epochs}/wesad.h5"

        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                model_path,
                monitor="val_loss",  # The metric name to monitor
                save_best_only=True,  # If True, it only saves the "best" model according to the quantity monitored
            ),
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10),
        ]

        history = model.fit(
            X_train,
            y_train,
            epochs=num_epochs,
            batch_size=50,
            verbose=1,
            class_weight={
                0: 1,
                1: 1,  # if weight_balance is None else weight_balance,
            },  # to address the imbalance of the class labels
            callbacks=callbacks,
            validation_data=(X_test, y_test),
        )

        print(f"Save model to: {model_path}")


def evaluate(
    gan_scores_acc,
    gan_scores_f1,
    gan_scores_precision,
    gan_scores_recall,
    all_subjects_X,
    all_subjects_y,
    data_type: DataType,
    num_epochs: int,
    num_subjects: int = 1,
    with_loso: bool = True,
    subject_ids=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17],
    additional_name: str = None,
):
    model_path = ""

    all_subjects_X_os = all_subjects_X
    all_accuracies = []
    all_precisions = []
    all_recalls = []
    all_f1s = []
    num_output_class = 2

    if not with_loso:
        print("*** Train on Synth, test on Real ***")
    else:
        print("*** Evaluate using 'Leave One Subject Out'-Method ***")

    if not with_loso:
        print(f"DATATYPE: {data_type}")
        if data_type == DataType.DGAN:
            model_path = (
                f"models/stress_detector/tstr/syn/dgan_30000/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_LSTM:
            model_path = (
                f"models/stress_detector/tstr/syn/cgan/no_dp/lstm/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_FCN:
            model_path = (
                f"models/stress_detector/tstr/syn/cgan/no_dp/fcn/{num_epochs}/wesad.h5"
            )
        if data_type == DataType.CGAN_TRANSFORMER:
            model_path = f"models/stress_detector/tstr/syn/cgan/no_dp/transformer/{num_epochs}/wesad.h5"
        if data_type == DataType.DPCGAN:
            model_path = f"models/stress_detector/tstr/syn/cgan/dp/{num_epochs}/{additional_name}_wesad.h5"
        if data_type == DataType.TIMEGAN:
            model_path = (
                f"models/stress_detector/tstr/syn/timegan/{num_epochs}/wesad.h5"
            )
        model = tf.keras.models.load_model(model_path)
        print(f"LOADED: {model_path}")

    all_confusion_matrices = []
    for i, subject_id in enumerate(subject_ids):
        test_index = constants.SUBJECT_IDS.index(subject_id)
        X_test = np.asarray(all_subjects_X_os[test_index])
        y_test = np.asarray(all_subjects_y[test_index])
        y_test = tf.keras.utils.to_categorical(y_test, num_output_class)

        if with_loso:
            if data_type == DataType.REAL:
                model_path = f"models/stress_detector/real/{num_epochs}/wesad_s{subject_id}.h5"  # Path to save the model file
            if data_type == DataType.DGAN:
                model_path = f"models/stress_detector/loso/syn/dgan_30000/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_LSTM:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/lstm/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_FCN:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/fcn/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.CGAN_TRANSFORMER:
                model_path = f"models/stress_detector/loso/syn/cgan/no_dp/transformer/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.DPCGAN:
                model_path = f"models/stress_detector/loso/syn/cgan/dp/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"
            if data_type == DataType.TIMEGAN:
                model_path = f"models/stress_detector/loso/syn/timegan/{num_epochs}_epochs/{num_subjects}_subjects/wesad_s{subject_id}.h5"

            print("MODEL_PATH:", model_path)
            model = tf.keras.models.load_model(model_path)

        y_pred = model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_true = np.argmax(y_test, axis=1)

        ## Create and store confusion matrix
        confusion = confusion_matrix(y_true, y_pred)
        all_confusion_matrices.append(confusion)

        evaluation_metrics = model.evaluate(X_test, y_test, verbose=0)

        accuracy = evaluation_metrics[1]
        precision = evaluation_metrics[2]
        recall = evaluation_metrics[3]

        f1 = 2 * precision * recall / (precision + recall)
        all_accuracies.append(accuracy)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    print(f"GAN: {data_type.name}")
    print(f"Evaluation of CNN model trained on {num_epochs} epochs\n")
    print(f"Subject\t\t Accuracy\tPrecision\tRecall\t\tF1-Score")
    print("************************************************************************")
    for i in range(len(all_accuracies)):
        print(
            f"S{subject_ids[i]}\t\t {round(all_accuracies[i], 5):.5f}\t{round(all_precisions[i], 5):.5f}\t\t{round(all_recalls[i], 5):.5f}\t\t{round(all_f1s[i], 5):.5f}"
        )

    print("************************************************************************")
    print(
        f"Average\t\t {round(np.mean(all_accuracies), 5):.5f}\t{round(np.mean(all_precisions), 5):.5f}\t\t{round(np.mean(all_recalls), 5):.5f}\t\t{round(np.mean(all_f1s), 5):.5f}\n\n\n"
    )

    key_suffix = "_aug" if with_loso else "_tstr"
    gan_scores_acc[f"{data_type.name}{key_suffix}{additional_name}"] = all_accuracies
    gan_scores_f1[f"{data_type.name}{key_suffix}{additional_name}"] = all_f1s
    gan_scores_precision[
        f"{data_type.name}{key_suffix}{additional_name}"
    ] = all_precisions
    gan_scores_recall[f"{data_type.name}{key_suffix}{additional_name}"] = all_recalls

    return {
        "acc": gan_scores_acc,
        "f1": gan_scores_f1,
        "precision": gan_scores_precision,
        "recall": gan_scores_recall,
    }
