import pyttsx3
import os

def generar_audios(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    comandos = {
        "si": "Esto es un ejemplo de comando para decir sí.",
        "no": "Este es un ejemplo de comando para decir no.",
        "continuar": "Esto es un ejemplo de comando para continuar."
    }
    
    engine = pyttsx3.init()
    for comando, texto in comandos.items():
        output_path = os.path.join(output_dir, f"{comando}.wav")
        engine.save_to_file(texto, output_path)
        print(f"Generado: {output_path}")
    
    engine.runAndWait()

# Directorio donde se generarán los audios
output_dir = "./audios_prueba"
generar_audios(output_dir)