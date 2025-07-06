from bruit.train_model.cnn import CNNClassifier
from bruit.train_model.cnn_seq import CNNSequenceClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from bruit.modules.config_loader import resolve_path, load_config
from bruit.modules.plotting import plot_training_history, report_metrics
import tensorflow as tf
from pathlib import Path
import numpy as np
import argparse
import logging
import datetime
import yaml
import io
import sys

def get_model_class(model_type):
    if model_type == "CNN":
        return CNNClassifier
    elif model_type == "CNN_Seq":
        return CNNSequenceClassifier
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def log_model_summary(model, logger):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + "\n"))
    summary_str = stream.getvalue()
    stream.close()
    logger.info("\n" + summary_str)

def load_data(npz_path, training_feature="mfcc"):
    data = np.load(npz_path, allow_pickle=True)
    X = np.stack(data[training_feature])  # or mel/zcr
    y = data["labels"]
    return X, y

def prepare_data(X, y):
    X = X[..., np.newaxis]  # (N, H, W, 1)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return train_test_split(X, y_encoded, stratify=y_encoded, test_size=0.2, random_state=42), le.classes_

def run_training(input_file, quiet=False):
    path_config = load_config("configs/paths.yaml")
    config = load_config("configs/parameters.yaml")

    if not input_file:
        input_file = resolve_path(path_config.trial_model_path)

    model_type = config.model_training.model_type

    log_path = resolve_path(path_config.log_path)
    log_path = log_path / model_type
    log_path.mkdir(parents=True, exist_ok=True)

    metadata_path = resolve_path(path_config.metadata_path)
    metadata_path = metadata_path / model_type
    metadata_path.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(path_config.model_path)
    model_path = model_path / model_type
    model_path.mkdir(parents=True, exist_ok=True)

    figure_path = resolve_path(path_config.figure_path)
    figure_path = figure_path / model_type
    figure_path.mkdir(parents=True, exist_ok=True)

    log_time = datetime.datetime.now().strftime("%H-%M-%S")
    # Logging setup
    log_file = log_path / f"model_training.log"
    logging.basicConfig(
                filename=log_file,
                filemode="w",
                level=logging.INFO,
                format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    sys.stdout = open(log_file, 'a')
    sys.stderr = sys.stdout
    tf.get_logger().setLevel('INFO')
    tf.get_logger().addHandler(logging.FileHandler(log_file))
    logger = logging.getLogger("bruit")
    if not quiet:
        logger.info(f"📁 Logging all preprocessing steps to {log_path}")

    input_folder = Path(input_file)
    npz_file = input_folder / f"{config.preprocessing.file_name}.{config.preprocessing.file_extension}"


    params = {
        "training_feature": config.model_training.training_feature, 
        "loss_function": config.model_training.loss_function,
        "epochs": config.model_training.epochs,
        "batch_size": config.model_training.batch_size,
        "learning_rate": config.model_training.learning_rate,
        "patience": config.model_training.patience,
        "validation_split": config.model_training.validation_split,
        "class_weights": config.model_training.class_weights,
        "model_name": config.model_training.model_name,
        "model_type": config.model_training.model_type,
        "model_path": model_path,
        "architecture": config.model_training.architecture
    }

    X, y = load_data(npz_file, training_feature=params["training_feature"])
    (X_train, X_val, y_train, y_val), class_names = prepare_data(X, y)

    print("Calculating class weights...")
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: w for i, w in enumerate(class_weights)}

    print("Building model...")
    ModelClass = get_model_class(model_type)
    clf = ModelClass(input_shape=X_train.shape[1:], num_classes=len(class_names), params=params)
    clf.compile()

    print("Training...")
    history = clf.fit(X_train, y_train, X_val, y_val, epochs=params["epochs"], batch_size=params["batch_size"], patience=params['patience'], class_weight=class_weight_dict)
    plot_training_history(history, figure_path / f"{params['model_name']}_training_history.png")

    print("Saving model...")
    clf.save(model_path)

    logger.info(f"Model saved to {model_path}")
    logger.info("Model Summary...")
    log_model_summary(clf.model, logger)

    # Save metadata
    total_time = datetime.datetime.now() - datetime.datetime.strptime(log_time, "%H-%M-%S").replace(
        year=datetime.datetime.now().year,
        month=datetime.datetime.now().month,
        day=datetime.datetime.now().day
    )
    logger.info(f"Total training time: {total_time}")
    params['total_training_time'] = str(total_time)
    yaml_path = metadata_path / f"{config.model_training.model_name}_params.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(params, f)
    logger.info("Training complete")

    # Report metrics
    report_metrics(clf, X_val, y_val, class_names, figure_path / f"{params['model_name']}_confusion-matrix.png",figure_path / f"{params['model_name']}_classification_report.json")

   
