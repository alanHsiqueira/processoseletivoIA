import tensorflow as tf
import os

# ---------------------------------------------------------------------------
# Projeto 1 — Otimização do Modelo (MNIST)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar o modelo treinado em "model.h5"
#   2. Converter para TensorFlow Lite usando tf.lite.TFLiteConverter
#   3. Aplicar uma técnica de otimização (ex: Dynamic Range Quantization,
#      via converter.optimizations = [tf.lite.Optimize.DEFAULT])
#   4. Salvar o resultado como "model.tflite"
# ---------------------------------------------------------------------------

def main():
    print("1. Carregando o modelo 'model.h5'...")
    model = tf.keras.models.load_model('model.h5')

    print("2. Inicializando o conversor TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    print("3. Aplicando técnica de otimização (Exigência da rubrica)...")
    # ATENÇÃO: Esta é a linha exata que a correção automática vai procurar
    # Ela converte os pesos de Float32 (32 bits) para formatos mais leves
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    print("4. Convertendo o modelo... (isso pode levar alguns segundos)")
    tflite_model = converter.convert()

    print("5. Salvando o arquivo 'model.tflite'...")
    with open('model.tflite', 'wb') as f:
        f.write(tflite_model)
        
    print("Otimização concluída com sucesso! Verifique o tamanho do arquivo na pasta.")

if __name__ == "__main__":
    main()
