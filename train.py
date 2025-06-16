# Importa as funções dos outros arquivos
from preprocess import load_and_preprocess_data
from model import create_cnn_model

# --- 1. Configurações ---
# Substitua pelo caminho onde você extraiu o dataset.
DATASET_BASE_DIR = '/home/viner/UFMS/IA/projeto_fraturas/archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15 # Número de épocas para o treinamento
MODEL_SAVE_PATH = 'fracture_detection_model.keras'

# --- 2. Carregar Dados ---
train_ds, val_ds, test_ds, class_names = load_and_preprocess_data(
    base_dir=DATASET_BASE_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- 3. Criar e Treinar o Modelo ---
model = create_cnn_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

print("\nIniciando o treinamento do modelo...")
history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds
)

# --- 4. Salvar o Modelo Treinado ---
print(f"\nTreinamento concluído. Salvando o modelo em: {MODEL_SAVE_PATH}")
model.save(MODEL_SAVE_PATH)

print("Processo finalizado.")