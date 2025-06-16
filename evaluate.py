import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Importa a nossa função de pré-processamento
from preprocess import load_and_preprocess_data

# --- 1. Configurações ---
# Garanta que estes caminhos e parâmetros sejam os mesmos do script de treinamento
DATASET_BASE_DIR = '/home/viner/UFMS/IA/projeto_fraturas/archive/Bone_Fracture_Binary_Classification/Bone_Fracture_Binary_Classification'
MODEL_PATH = 'fracture_detection_model.keras'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# --- 2. Carregar o Modelo e os Dados de Teste ---
print(f"Carregando modelo de: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)

print("Carregando e preparando dados de teste...")
# Usamos a função do nosso primeiro script para carregar apenas os dados de teste
_, _, test_ds, class_names = load_and_preprocess_data(
    base_dir=DATASET_BASE_DIR,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
print(f"Classes identificadas: {class_names}") # Deve ser ['fractured', 'non_fractured'] ou similar

# --- 3. Avaliação Geral do Modelo ---
print("\nAvaliando o desempenho geral do modelo no conjunto de teste...")
loss, accuracy = model.evaluate(test_ds)
print(f"Acurácia no Teste: {accuracy:.2f}")
print(f"Perda no Teste: {loss:.2f}")

# --- 4. Relatório de Classificação e Matriz de Confusão ---
# Para gerar o relatório e a matriz, precisamos fazer previsões em todo o conjunto de teste
y_pred_prob = model.predict(test_ds)
y_pred = (y_pred_prob > 0.5).astype(int).flatten() # Converte probabilidades em classes (0 ou 1)
y_true = np.concatenate([y for x, y in test_ds], axis=0).astype(int)

# Relatório de Classificação (Precisão, Recall, F1-Score)
print("\nRelatório de Classificação:")
print(classification_report(y_true, y_pred, target_names=class_names))

# Matriz de Confusão
print("\nMatriz de Confusão:")
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Previsão do Modelo')
plt.ylabel('Rótulo Verdadeiro')
plt.title('Matriz de Confusão')
plt.show()

# --- 5. Análise de Exemplos (Acertos e Erros) ---
# Mostra alguns exemplos para análise qualitativa, como solicitado no trabalho 
print("\nAnalisando exemplos de previsões...")
plt.figure(figsize=(15, 15))

# Pega um lote de imagens do conjunto de teste
for images, labels in test_ds.take(1):
    predictions = model.predict(images)
    for i in range(min(16, len(images))): # Mostra até 16 imagens
        ax = plt.subplot(4, 4, i + 1)
        
        # Inverte a normalização para visualização
        img_array = images[i].numpy()
        
        plt.imshow(img_array)
        plt.axis("off")
        
        true_label = class_names[int(labels[i])]
        pred_prob = predictions[i][0]
        pred_label = class_names[1] if pred_prob > 0.5 else class_names[0]
        
        # Define a cor do título: verde para acerto, vermelho para erro
        title_color = 'green' if pred_label == true_label else 'red'
        
        plt.title(f"Verdadeiro: {true_label}\nPrevisto: {pred_label} ({pred_prob:.2f})", color=title_color)

plt.tight_layout()
plt.show()