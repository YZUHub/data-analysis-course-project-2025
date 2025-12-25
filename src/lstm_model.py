# src/lstm_model.py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.regularizers import l2
import os

def build_lstm_model(input_shape, num_classes=12):
    """
    SIMPLE LSTM for 80% accuracy
    """
    print("Building Simple LSTM")
    print(f"   Input shape: {input_shape}")
    print(f"   Classes: {num_classes}")
    
    model = models.Sequential(name="Simple_LSTM")
    
    model.add(layers.LSTM(
        units=128,
        input_shape=input_shape,
        return_sequences=False 
    ))
    model.add(layers.Dropout(0.3))
    
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.2))
    
    # Output
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model