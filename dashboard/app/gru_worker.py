from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

import numpy as np

# Keep TensorFlow logs minimal in worker output.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
# Prefer CPU and conservative threading for stability in subprocess inference.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

import tensorflow as tf


class AttnPoolSum(tf.keras.layers.Layer):
    """Safe replacement for serialized attention Lambda reduce_sum."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs, mask=None):  # noqa: ARG002 - signature compatibility
        return tf.reduce_sum(inputs, axis=1)


def _prepare_loaded_gru_model(model: tf.keras.Model) -> tf.keras.Model:
    lambda_layer_type = tf.keras.layers.Lambda
    for layer in getattr(model, "layers", []):
        if not isinstance(layer, lambda_layer_type):
            continue
        fn = getattr(layer, "function", None)
        globals_dict = getattr(fn, "__globals__", None)
        if isinstance(globals_dict, dict):
            globals_dict.setdefault("tf", tf)

    def clone_fn(layer: tf.keras.layers.Layer) -> tf.keras.layers.Layer:
        if isinstance(layer, lambda_layer_type):
            return AttnPoolSum(name=layer.name)
        return layer.__class__.from_config(layer.get_config())

    try:
        cloned = tf.keras.models.clone_model(model, clone_function=clone_fn)
        cloned.set_weights(model.get_weights())
        return cloned
    except Exception:
        # Fall back to original model if cloning is not possible.
        return model


def _build_lambda_shape_patched_archive(model_path: Path) -> Path | None:
    with zipfile.ZipFile(model_path, "r") as archive:
        config = json.loads(archive.read("config.json"))
        changed = False

        for layer in config.get("config", {}).get("layers", []):
            if layer.get("class_name") != "Lambda":
                continue

            layer_config = layer.setdefault("config", {})
            if "output_shape" in layer_config:
                continue

            input_shape = layer.get("build_config", {}).get("input_shape")
            if not isinstance(input_shape, list) or not input_shape:
                continue

            last_dim = input_shape[-1]
            if not isinstance(last_dim, int):
                continue

            layer_config["output_shape"] = [last_dim]
            changed = True

        if not changed:
            return None

        with tempfile.NamedTemporaryFile(suffix=".keras", delete=False) as handle:
            patched_path = Path(handle.name)
        with zipfile.ZipFile(patched_path, "w") as patched:
            for member in archive.infolist():
                payload = archive.read(member.filename)
                if member.filename == "config.json":
                    payload = json.dumps(config).encode("utf-8")
                patched.writestr(member, payload)

        return patched_path


def _load_gru_model(model_path: Path) -> tf.keras.Model:
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return _prepare_loaded_gru_model(model)
    except ValueError as exc:
        message = str(exc)
        if "Lambda layer" not in message and "safe_mode=False" not in message:
            raise

        try:
            model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False)
            return _prepare_loaded_gru_model(model)
        except NotImplementedError as retry_exc:
            text = str(retry_exc).casefold()
            if "output_shape" not in text and "infer the shape" not in text:
                raise

            patched_path = _build_lambda_shape_patched_archive(model_path)
            if patched_path is None:
                raise

            try:
                model = tf.keras.models.load_model(patched_path, compile=False, safe_mode=False)
                return _prepare_loaded_gru_model(model)
            finally:
                patched_path.unlink(missing_ok=True)


def _disable_gpu_for_stability() -> None:
    # Metal-backed TensorFlow can crash with hard traps in some local environments.
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    _disable_gpu_for_stability()

    x_windows = np.load(input_path)
    model = _load_gru_model(model_path)

    x_array = np.asarray(x_windows, dtype=np.float32)
    input_name = model.inputs[0].name.split(":")[0] if getattr(model, "inputs", None) else None
    if input_name:
        outputs = model({input_name: x_array}, training=False)
    else:
        outputs = model(x_array, training=False)

    probs = np.asarray(outputs).reshape(-1).astype(float)
    np.save(output_path, probs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
