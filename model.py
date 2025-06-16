from tensorflow.keras import layers, models

def create_cnn_model(input_shape=(224, 224, 3)):
    """
    Cria e compila um modelo de Rede Neural Convolucional (CNN).

    Args:
        input_shape (tuple): A forma das imagens de entrada (altura, largura, canais).

    Returns:
        tensorflow.keras.Model: O modelo de CNN compilado.
    """
    model = models.Sequential([
        # Camada de Entrada
        layers.Input(shape=input_shape),

        # Primeiro Bloco Convolucional
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Segundo Bloco Convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Terceiro Bloco Convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),

        # Camadas de Classificação
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5), # Camada de Dropout para reduzir overfitting
        layers.Dense(1, activation='sigmoid') # Saída binária com ativação sigmoide
    ])

    # Compila o modelo
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    print("Modelo CNN criado e compilado com sucesso.")
    model.summary() # Imprime um resumo da arquitetura do modelo
    return model