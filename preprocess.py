import tensorflow as tf
import os

def load_and_preprocess_data(base_dir, img_size=(224, 224), batch_size=32):
    """
    Carrega, pré-processa e prepara os conjuntos de dados de treinamento, validação e teste.

    Args:
        base_dir (str): Caminho para a pasta principal do dataset.
        img_size (tuple): Tamanho para redimensionar as imagens.
        batch_size (int): Tamanho do lote.

    Returns:
        tuple: Contém os datasets de treinamento, validação, teste e os nomes das classes.
    """
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'val')
    test_dir = os.path.join(base_dir, 'test')

    # Carrega os datasets
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )

    validation_dataset = tf.keras.utils.image_dataset_from_directory(
        validation_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    class_names = train_dataset.class_names

    # Normaliza os pixels para o intervalo [0, 1]
    normalization_layer = tf.keras.layers.Rescaling(1./255)
    train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
    validation_dataset = validation_dataset.map(lambda x, y: (normalization_layer(x), y))
    test_dataset = test_dataset.map(lambda x, y: (normalization_layer(x), y))

    # Otimiza o pipeline de dados para performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

    print("Dados de pré-processamento carregados com sucesso.")
    return train_dataset, validation_dataset, test_dataset, class_names