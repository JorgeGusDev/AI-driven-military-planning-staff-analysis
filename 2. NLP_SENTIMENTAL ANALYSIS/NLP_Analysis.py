import os
import re
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuración del directorio de entrada
input_directory = r"C:\Users\LENOVO\OneDrive - UAM\Jorge G\7. IDEA_LAB\7. SNA & SA\4. PAPER_SNA_SA\1. Data_Technics & Background_Paper\2. NLP_SENTIMENTAL ANALYSIS\0. ROW CHATS\GROUPS CHAT (CICLO I)_ENERO A MAYO\CleanerChats"

# Cargar el modelo de análisis de sentimiento
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Función para mapear los resultados de sentimiento a categorías simples
def map_sentiment(label):
    if label in ["5 stars", "4 stars"]:
        return "positivo"
    elif label in ["1 star", "2 stars"]:
        return "negativo"
    else:
        return "neutro"

# Función para extraer el número de usuario entre corchetes
def extract_user_number(text):
    match = re.search(r'\[(\d+)\]', text)
    return match.group(1) if match else "Unknown"

# Procesar cada archivo en la carpeta
overall_results = []  # Lista para almacenar los resultados generales

for filename in tqdm(os.listdir(input_directory), desc="Procesando archivos"):
    if filename.endswith('.txt'):
        file_path = os.path.join(input_directory, filename)
        
        # Lista para almacenar los resultados de cada línea
        results = []
        
        # Leer y analizar cada línea del archivo de texto con barra de progreso para las líneas
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            for line in tqdm(lines, desc=f"Procesando líneas en {filename}", leave=False):
                line = line.strip()
                if line:  # Evitar líneas vacías
                    user_number = extract_user_number(line)
                    sentiment = sentiment_pipeline(line)
                    sentiment_label = sentiment[0]['label']
                    sentiment_score = sentiment[0]['score']
                    
                    # Mapear el sentimiento a las categorías simplificadas
                    simple_sentiment = map_sentiment(sentiment_label)
                    
                    results.append({
                        'user': user_number,
                        'phrase': line,
                        'sentiment_label': simple_sentiment,
                        'sentiment_score': sentiment_score,
                        'original_label': sentiment_label  # Guardar también la etiqueta original de estrellas
                    })
                    overall_results.append({
                        'filename': filename,
                        'user': user_number,
                        'sentiment_label': simple_sentiment
                    })
        
        # Convertir los resultados en un DataFrame
        sentiment_results = pd.DataFrame(results)
        
        # Generar el nombre del archivo de salida
        output_excel_path = os.path.join(input_directory, f"{os.path.splitext(filename)[0]}_sentiment_analysis.xlsx")
        
        # Guardar en un archivo Excel con las etiquetas originales y simplificadas
        sentiment_results.to_excel(output_excel_path, index=False)
        print(f"Resultados guardados en '{output_excel_path}'")
        
        # Generar el gráfico de barras para cada usuario en el archivo actual
        plt.figure(figsize=(10, 6))
        sns.countplot(data=sentiment_results, x='user', hue='sentiment_label', palette="viridis")
        plt.title(f"Distribución de Sentimientos por Usuario en {filename}")
        plt.xlabel("Usuario")
        plt.ylabel("Cantidad de Sentimientos")
        plt.legend(title="Sentimiento")
        plt.savefig(os.path.join(input_directory, f"{os.path.splitext(filename)[0]}_user_sentiment_distribution.png"))
        plt.close()

# Crear un DataFrame general con todos los resultados
overall_df = pd.DataFrame(overall_results)

# Generar un gráfico de comparación general entre archivos
plt.figure(figsize=(12, 6))
sns.countplot(data=overall_df, x='filename', hue='sentiment_label', palette="viridis")
plt.title("Comparación de Sentimientos entre Archivos")
plt.xlabel("Archivo")
plt.ylabel("Cantidad de Sentimientos")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Sentimiento")
plt.tight_layout()
plt.savefig(os.path.join(input_directory, "sentiment_comparison_across_files.png"))
plt.close()

print("Análisis y gráficos completados para todos los archivos.")
