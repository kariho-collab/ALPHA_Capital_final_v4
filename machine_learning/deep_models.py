"""
04. Deep Learning: LSTM & Temporal Transformer Experiments
Implements LSTM and Self-Attention Transformer architectures in Keras/TensorFlow.
"""
import logging
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

# TensorFlow / Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


class SequenceDataGenerator:
    @staticmethod
    def create_sliding_windows(
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int = 24
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transforms 2D tabular array (T, D) into 3D sequential tensors (T - seq_len, seq_len, D).
        """
        T, D = X.shape
        num_samples = T - seq_len
        X_seq = np.zeros((num_samples, seq_len, D), dtype=np.float32)
        y_seq = np.zeros(num_samples, dtype=np.float32)

        for i in range(num_samples):
            X_seq[i] = X[i : i + seq_len]
            y_seq[i] = y[i + seq_len]

        return X_seq, y_seq


class DeepLearningExperimenter:
    def __init__(self, seq_len: int = 24, random_state: int = 42):
        self.seq_len = seq_len
        self.random_state = random_state
        tf.random.set_seed(random_state)

    def build_lstm_model(self, input_shape: Tuple[int, int]) -> keras.Model:
        """
        Builds a stacked LSTM with recurrent dropout and residual projection.
        """
        inputs = layers.Input(shape=input_shape)
        x = layers.LSTM(64, return_sequences=True, dropout=0.2)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(32, return_sequences=False, dropout=0.2)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="Forex_LSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def build_transformer_model(
        self,
        input_shape: Tuple[int, int],
        head_size: int = 32,
        num_heads: int = 4,
        ff_dim: int = 64
    ) -> keras.Model:
        """
        Builds a Temporal Multi-Head Attention Transformer for sequential pattern discovery.
        """
        inputs = layers.Input(shape=input_shape)
        # 1. Multi-Head Self-Attention Block
        attn_output = layers.MultiHeadAttention(
            key_dim=head_size, num_heads=num_heads, dropout=0.1
        )(inputs, inputs)
        x = layers.Add()([inputs, attn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # 2. Feed-Forward Block
        ffn = keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dropout(0.1),
            layers.Dense(input_shape[-1]),
        ])
        ffn_output = ffn(x)
        x = layers.Add()([x, ffn_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)

        # 3. Global Pooling & Classification Head
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name="Forex_Transformer")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
            metrics=["AUC"]
        )
        return model

    def train_and_evaluate_deep_models(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str = "target_long",
        train_ratio: float = 0.75,
        epochs: int = 10,
        batch_size: int = 64
    ) -> Dict[str, Any]:
        """
        Prepares 3D sequence tensors, trains LSTM and Transformer, and benchmarks out-of-sample AUC.
        """
        X = df[feature_cols].values
        y = df[target_col].values

        split_idx = int(len(X) * train_ratio)
        scaler = StandardScaler()
        X_train_raw = scaler.fit_transform(X[:split_idx])
        X_test_raw = scaler.transform(X[split_idx:])

        # Generate sequential 3D tensors
        X_train_seq, y_train_seq = SequenceDataGenerator.create_sliding_windows(
            X_train_raw, y[:split_idx], self.seq_len
        )
        X_test_seq, y_test_seq = SequenceDataGenerator.create_sliding_windows(
            X_test_raw, y[split_idx:], self.seq_len
        )

        input_shape = (self.seq_len, len(feature_cols))
        logger.info(f"Training Deep Learning Models: Input Shape = {input_shape}, Train Samples = {len(X_train_seq):,}...")

        # 1. Train LSTM
        lstm_model = self.build_lstm_model(input_shape)
        lstm_model.fit(
            X_train_seq, y_train_seq,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        p_lstm = lstm_model.predict(X_test_seq, verbose=0).flatten()

        # 2. Train Transformer
        trans_model = self.build_transformer_model(input_shape)
        trans_model.fit(
            X_train_seq, y_train_seq,
            validation_split=0.15,
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        p_trans = trans_model.predict(X_test_seq, verbose=0).flatten()

        results = {
            "lstm": {
                "test_auc": float(roc_auc_score(y_test_seq, p_lstm)),
                "brier_score": float(brier_score_loss(y_test_seq, p_lstm)),
                "probabilities": p_lstm
            },
            "transformer": {
                "test_auc": float(roc_auc_score(y_test_seq, p_trans)),
                "brier_score": float(brier_score_loss(y_test_seq, p_trans)),
                "probabilities": p_trans
            },
            "y_test_seq": y_test_seq
        }

        logger.info(f"Deep Learning OOS AUC: LSTM = {results['lstm']['test_auc']:.3f}, Transformer = {results['transformer']['test_auc']:.3f}")
        return results
