import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import speech_recognition as sr
import librosa
import os

# Definición de variables difusas
# Variables de entrada: duración, energía y contexto
duracion = ctrl.Antecedent(np.arange(0, 11, 1), 'duracion')
energia = ctrl.Antecedent(np.arange(0, 11, 1), 'energia')
contexto = ctrl.Antecedent(np.arange(0, 11, 1), 'contexto')  # Memoria previa o estado

# Variable de salida: comando
comando = ctrl.Consequent(np.arange(0, 11, 1), 'comando')

# Funciones de membresía para duración
duracion['corta'] = fuzz.trimf(duracion.universe, [0, 0, 5])
duracion['media'] = fuzz.trimf(duracion.universe, [0, 5, 10])
duracion['larga'] = fuzz.trimf(duracion.universe, [5, 10, 10])

# Funciones de membresía para energía
energia['baja'] = fuzz.trimf(energia.universe, [0, 0, 5])
energia['media'] = fuzz.trimf(energia.universe, [0, 5, 10])
energia['alta'] = fuzz.trimf(energia.universe, [5, 10, 10])

# Funciones de membresía para contexto
contexto['neutro'] = fuzz.trimf(contexto.universe, [0, 0, 5])
contexto['activo'] = fuzz.trimf(contexto.universe, [5, 10, 10])

# Funciones de membresía para comandos
comando['si'] = fuzz.trimf(comando.universe, [0, 0, 5])
comando['no'] = fuzz.trimf(comando.universe, [0, 5, 10])
comando['continuar'] = fuzz.trimf(comando.universe, [5, 10, 10])

# Definición de reglas difusas
rule1 = ctrl.Rule(duracion['corta'] & energia['baja'] & contexto['neutro'], comando['si'])
rule2 = ctrl.Rule(duracion['corta'] & energia['media'] & contexto['neutro'], comando['no'])
rule3 = ctrl.Rule(duracion['media'] & energia['alta'] & contexto['activo'], comando['continuar'])
rule4 = ctrl.Rule(duracion['larga'] & energia['alta'] & contexto['activo'], comando['continuar'])

# Añadir una regla genérica para manejar casos no definidos
rule_default = ctrl.Rule(~(duracion['corta'] | duracion['media'] | duracion['larga']) |
                         ~(energia['baja'] | energia['media'] | energia['alta']) |
                         ~(contexto['neutro'] | contexto['activo']),
                         comando['si'])  # Regla por defecto

# Controlador difuso
speech_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule_default])
speech_simulation = ctrl.ControlSystemSimulation(speech_ctrl)

# Función para extraer características del audio
def extraer_caracteristicas(audio_path):
    signal, sr = librosa.load(audio_path)
    duracion = librosa.get_duration(y=signal, sr=sr)
    energia = np.mean(librosa.feature.rms(y=signal)) * 10  # Escalado simple
    return duracion, energia

# Función para clasificar comandos con memoria
def clasificar_comando(duracion_input, energia_input, contexto_input):
    # Asegurar que las entradas están en rango
    duracion_input = np.clip(duracion_input, 0, 10)
    energia_input = np.clip(energia_input, 0, 10)
    contexto_input = np.clip(contexto_input, 0, 10)

    # Asignar valores
    speech_simulation.input['duracion'] = duracion_input
    speech_simulation.input['energia'] = energia_input
    speech_simulation.input['contexto'] = contexto_input
    
    # Procesar
    try:
        speech_simulation.compute()
        resultado = speech_simulation.output['comando']
    except KeyError:
        resultado = "Sin resultado"
    return resultado

# Reconocimiento de texto a partir de audio
def reconocer_texto(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        texto = recognizer.recognize_google(audio, language='es-ES')
        return texto
    except sr.UnknownValueError:
        return "No se pudo reconocer"

# Directorio con audios de prueba
audio_dir = "./audios_prueba"  # Reemplazar con el path a tus audios

# Procesar cada audio en el directorio
contexto_anterior = 5  # Inicialmente neutro
for audio_file in os.listdir(audio_dir):
    if audio_file.endswith(".wav"):
        audio_path = os.path.join(audio_dir, audio_file)
        print(f"Procesando archivo: {audio_file}")
        
        # Extraer características
        duracion, energia = extraer_caracteristicas(audio_path)
        print(f"Duración: {duracion:.2f}, Energía: {energia:.2f}")
        
        # Clasificar usando lógica difusa con memoria previa
        resultado_difuso = clasificar_comando(duracion, energia, contexto_anterior)
        print(f"Resultado difuso: {resultado_difuso}")
        
        # Reconocimiento de texto
        texto = reconocer_texto(audio_path)
        print(f"Texto reconocido: {texto}")
        
        # Actualizar contexto basado en la salida difusa
        contexto_anterior = resultado_difuso
        print("---")
