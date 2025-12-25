import tensorflow as tf
from tensorflow.keras import layers, models

def build_simple_cnn(input_shape, num_classes):
    print(f"   Input shape: {input_shape}")
    print(f"   Classes: {num_classes}")
    
    model = models.Sequential(name="CNN_3Layer_80")
    
    # Conv Block 1
    model.add(layers.Conv1D(
        filters=64, 
        kernel_size=5, 
        activation='relu',
        padding='same',
        input_shape=input_shape,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.4))

    # Conv Block 2
    model.add(layers.Conv1D(
        filters=128, 
        kernel_size=3, 
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Dropout(0.5))
    
    # Conv Block 3 
    model.add(layers.Conv1D(
        filters=256, 
        kernel_size=3, 
        activation='relu',
        padding='same',
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.GlobalAveragePooling1D())
    model.add(layers.Dropout(0.6))
    
    # Classifier
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    # Lower learning rate
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=0.0005,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    return model