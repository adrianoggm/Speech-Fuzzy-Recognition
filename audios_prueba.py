import pyttsx3
import os

def generar_audios(output_dir):
    """
    Genera audios .wav con diferentes estados de ánimo (animado o triste).
    Ajusta la tasa de habla (rate) y el texto para reforzar diferencias
    en 'pitch' y 'energy' a fin de que un clasificador difuso
    pueda distinguir mejor entre 'animado' y 'triste'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Diccionario con el nombre del archivo, texto a decir y estado anímico
    # Se han ajustado los textos para enfatizar diferencias entonativas.
    comandos = {
        "animado_si": (
            "¡Sí, sí, sí! ¡Estoy muy contento hoy! ¡Qué energía tan positiva!", 
            "animada"
        ),
        "animado_no": (
            "¡No, pero no pasa nada! Sigamos adelante con mucho ánimo y ganas.", 
            "animada"
        ),
        "animado_continuar": (
            "¡Claro que sí! Continuemos con entusiasmo y alegría. ¡Me encanta hablar!", 
            "animada"
        ),
        "triste_si": (
            "Bueno... creo que sí... aunque me siento un poco desanimado.", 
            "triste"
        ),
        "triste_no": (
            "No... en realidad no me apetece mucho. Lo siento, pero hoy estoy de bajón.", 
            "triste"
        ),
        "triste_continuar": (
            "Podemos continuar... aunque la verdad no encuentro muchas ganas ahora mismo.", 
            "triste"
        )
    }
    
    engine = pyttsx3.init()
    
    for nombre_archivo, (texto, estado_animo) in comandos.items():
        output_path = os.path.join(output_dir, f"{nombre_archivo}.wav")
        
        # Ajustar la velocidad (rate) y, opcionalmente, la voz.
        # 'animada' -> más rápido, 'triste' -> más lento.
        if estado_animo == "animada":
            engine.setProperty('rate', 190)  # algo más rápido de lo normal
        else:
            engine.setProperty('rate', 120)  # más lento, simulando decaimiento
        
        # Ajustar el volumen en ambos casos
        engine.setProperty('volume', 1.0)  # volumen máximo
        
        # Generar el archivo de audio
        engine.save_to_file(texto, output_path)
        print(f"Generando [{estado_animo.upper()}]: {output_path} => '{texto}'")
    
    # Ejecutar la síntesis
    engine.runAndWait()

# Directorio donde se generarán los audios
if __name__ == "__main__":
    output_dir = "./audios_prueba"
    generar_audios(output_dir)