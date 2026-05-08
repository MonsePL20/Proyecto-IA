
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

img_size = (128, 128)
batch_size = 32


train_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "dataset",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

# ============================================
# NOMBRES DE CLASES
# ============================================

class_names = train_ds.class_names

print("\nCLASES DETECTADAS:")
print(class_names)

# ============================================
# NORMALIZACIÓN
# ============================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

val_ds = val_ds.map(
    lambda x, y: (normalization_layer(x), y)
)

# ============================================
# OPTIMIZACIÓN DE RENDIMIENTO
# ============================================

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# ============================================
# MODELO CNN
# ============================================

model = tf.keras.models.Sequential([

    # CAPA CONVOLUCIONAL 1
    tf.keras.layers.Conv2D(
        32,
        (3, 3),
        activation='relu',
        input_shape=(128, 128, 3)
    ),

    tf.keras.layers.MaxPooling2D(),

    # CAPA CONVOLUCIONAL 2
    tf.keras.layers.Conv2D(
        64,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    # CAPA CONVOLUCIONAL 3
    tf.keras.layers.Conv2D(
        128,
        (3, 3),
        activation='relu'
    ),

    tf.keras.layers.MaxPooling2D(),

    # APLANAMIENTO
    tf.keras.layers.Flatten(),

    # CAPA DENSA
    tf.keras.layers.Dense(
        128,
        activation='relu'
    ),

    # CAPA DE SALIDA
    tf.keras.layers.Dense(
        8,
        activation='softmax'
    )
])

# ============================================
# RESUMEN DEL MODELO
# ============================================

model.summary()

# ============================================
# COMPILACIÓN
# ============================================

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# ============================================
# EARLY STOPPING
# ============================================

early_stop = tf.keras.callbacks.EarlyStopping(
    patience=3,
    restore_best_weights=True
)

# ============================================
# ENTRENAMIENTO
# ============================================

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# ============================================
# EVALUACIÓN
# ============================================

loss, acc = model.evaluate(val_ds)

print("\nPRECISIÓN FINAL DEL MODELO:")
print(acc)

# ============================================
# GRÁFICA DE ENTRENAMIENTO
# ============================================

plt.figure(figsize=(8,5))

plt.plot(
    history.history['accuracy'],
    label='Entrenamiento',
    color='green'
)

plt.plot(
    history.history['val_accuracy'],
    label='Validación',
    color='limegreen'
)

plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')

plt.legend()

# Guardar gráfica
plt.savefig("grafica_entrenamiento.png")

plt.show()


y_true = []
y_pred = []

for images, labels in val_ds:

    preds = model.predict(images)

    preds = np.argmax(preds, axis=1)

    y_pred.extend(preds)

    y_true.extend(labels.numpy())

# ============================================
# MATRIZ DE CONFUSIÓN
# ============================================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

disp.plot(cmap='Greens')

plt.title("Matriz de Confusión")

# Guardar matriz
plt.savefig("matriz_confusion.png")

plt.show()

# ============================================
# REPORTE DE CLASIFICACIÓN
# ============================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True
)

print("\nREPORTE DE CLASIFICACIÓN:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names
    )
)

# ============================================
# PRECISIÓN POR CLASE
# ============================================

precision_por_clase = {}

for clase in class_names:

    precision_por_clase[clase] = report[clase]['precision']

# ============================================
# GRÁFICA DE PRECISIÓN POR CLASE
# ============================================

plt.figure(figsize=(10,5))

plt.bar(
    precision_por_clase.keys(),
    precision_por_clase.values(),
    color='forestgreen',
    edgecolor='black'
)

plt.xticks(rotation=45)

plt.title("Precisión por Clase")

plt.ylabel("Precisión")

# Guardar gráfica
plt.savefig("precision_por_clase.png")

plt.show()

# ============================================
# GUARDAR MODELO
# ============================================

model.save("modelo_perros_gatos.h5")

print("\nMODELO GUARDADO CORRECTAMENTE")

