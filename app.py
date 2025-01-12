import os
import numpy as np
import librosa
import skfuzzy as fuzz
from skfuzzy import control as ctrl

###############################################################################
# 1. DEFINIMOS EL SISTEMA DIFUSO (pitch, energy, q_mem) -> emocion (animada/triste)
###############################################################################

# Variables difusas (entradas)
pitch = ctrl.Antecedent(np.arange(0, 501, 1), 'pitch')
energy = ctrl.Antecedent(np.arange(0, 11, 1), 'energy')
q_mem = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'q_mem')

# Variable difusa (salida)
emocion = ctrl.Consequent(np.arange(0, 11, 1), 'emocion')

# Definición de funciones de membresía
pitch['bajo'] = fuzz.trimf(pitch.universe, [0, 0, 150])
pitch['alto'] = fuzz.trimf(pitch.universe, [150, 500, 500])

energy['baja'] = fuzz.trimf(energy.universe, [0, 0, 5])
energy['alta'] = fuzz.trimf(energy.universe, [5, 10, 10])

q_mem['LOW']  = fuzz.trimf(q_mem.universe, [0, 0, 0.5])
q_mem['HIGH'] = fuzz.trimf(q_mem.universe, [0.5, 1, 1])

emocion['triste']  = fuzz.trimf(emocion.universe, [0, 0, 5])
emocion['animada'] = fuzz.trimf(emocion.universe, [5, 10, 10])

# Reglas difusas
rule_emo1 = ctrl.Rule(
    pitch['bajo'] & energy['baja'] & q_mem['LOW'], 
    emocion['triste']
)
rule_emo2 = ctrl.Rule(
    pitch['alto'] & energy['alta'] & q_mem['HIGH'], 
    emocion['animada']
)
rule_emo3 = ctrl.Rule(
    pitch['alto'] & q_mem['LOW'], 
    emocion['triste']
)
rule_emo_def = ctrl.Rule(
    ~(pitch['bajo'] | pitch['alto']) |
    ~(energy['baja'] | energy['alta']) |
    ~(q_mem['LOW'] | q_mem['HIGH']),
    emocion['animada']
)

emo_ctrl = ctrl.ControlSystem([rule_emo1, rule_emo2, rule_emo3, rule_emo_def])
emo_simulation = ctrl.ControlSystemSimulation(emo_ctrl)

def classify_emotion(pitch_val, energy_val, q_val):
    """
    Clasifica la voz como 'animada' o 'triste' dada la pitch, energy y el estado difuso q_val.
    Retorna un valor numérico en [0..10].
    """
    emo_simulation.input['pitch'] = np.clip(pitch_val, 0, 500)
    emo_simulation.input['energy'] = np.clip(energy_val, 0, 10)
    emo_simulation.input['q_mem'] = np.clip(q_val, 0, 1)
    
    emo_simulation.compute()
    return emo_simulation.output['emocion']

###############################################################################
# 2. OPCIONAL: DEFINICIÓN BÁSICA DEL FLIP-FLOP DIFUSO (SR) PARA LA 'MEMORIA'
###############################################################################

Q_old = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'Q_old')
S = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'S')
R = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'R')
Q_new = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'Q_new')

Q_old['LOW']  = fuzz.trimf(Q_old.universe,  [0, 0, 0.5])
Q_old['HIGH'] = fuzz.trimf(Q_old.universe,  [0.5, 1, 1])
S['LOW']      = fuzz.trimf(S.universe,      [0, 0, 0.5])
S['HIGH']     = fuzz.trimf(S.universe,      [0.5, 1, 1])
R['LOW']      = fuzz.trimf(R.universe,      [0, 0, 0.5])
R['HIGH']     = fuzz.trimf(R.universe,      [0.5, 1, 1])
Q_new['LOW']  = fuzz.trimf(Q_new.universe,  [0, 0, 0.5])
Q_new['HIGH'] = fuzz.trimf(Q_new.universe,  [0.5, 1, 1])

rule_set   = ctrl.Rule(S['HIGH'] & R['LOW'],  Q_new['HIGH'])
rule_reset = ctrl.Rule(S['LOW']  & R['HIGH'], Q_new['LOW'])
rule_holdH = ctrl.Rule(Q_old['HIGH'] & S['LOW'] & R['LOW'], Q_new['HIGH'])
rule_holdL = ctrl.Rule(Q_old['LOW']  & S['LOW'] & R['LOW'], Q_new['LOW'])
rule_indet = ctrl.Rule(S['HIGH'] & R['HIGH'], Q_new['HIGH'])  # arbitraje

ff_control = ctrl.ControlSystem([rule_set, rule_reset, rule_holdH, rule_holdL, rule_indet])
ff_simulation = ctrl.ControlSystemSimulation(ff_control)

def update_flip_flop(q_old_val, s_val, r_val):
    ff_simulation.input['Q_old'] = np.clip(q_old_val, 0, 1)
    ff_simulation.input['S']     = np.clip(s_val, 0, 1)
    ff_simulation.input['R']     = np.clip(r_val, 0, 1)
    ff_simulation.compute()
    return ff_simulation.output['Q_new']

###############################################################################
# 3. EXTRACCIÓN DE CARACTERÍSTICAS (pitch, energy) PARA CLASIFICAR
###############################################################################

def extraer_caracteristicas(audio_path):
    """
    Carga un audio y retorna pitch medio y energy media (ambas escaladas).
    - Pitch estimado con librosa.pyin (requiere librosa >= 0.8).
    - Energy calculada como RMS normalizado (0..10 aproximado).
    """
    y, sr = librosa.load(audio_path)
    
    # 3.1 Cálculo de energy (RMS)
    rms = librosa.feature.rms(y=y)[0]  # array con rms en cada frame
    energy_val = np.mean(rms) * 10     # simple escalado [0..10 aprox]
    
    # 3.2 Cálculo de pitch (pyin)
    # Para evitar warnings, capta silencios y asigna NaN => promediamos ignorando NaN
    f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=50, fmax=400, sr=sr)
    if f0 is not None:
        pitch_val = np.nanmean(f0)  # media ignorando nan
        if np.isnan(pitch_val):
            pitch_val = 0.0
    else:
        pitch_val = 0.0
    
    # Ajustar pitch a [0..500]
    pitch_val = np.clip(pitch_val, 0, 500)
    
    return pitch_val, energy_val

###############################################################################
# 4. PROCESAR LOS AUDIOS Y CLASIFICAR COMO ANIMADOS O TRISTES
###############################################################################

def clasificar_audios(dir_audios):
    """
    Recorre cada .wav en dir_audios, extrae (pitch, energy),
    usa q_mem=0.0 (por ejemplo) y clasifica con la red difusa.
    """
    q_mem_anterior = 0.0  # si quisieras actualizar un flip-flop, podrías usarlo
    for fname in os.listdir(dir_audios):
        if fname.endswith(".wav"):
            path = os.path.join(dir_audios, fname)
            pitch_val, energy_val = extraer_caracteristicas(path)
            
            # (Opcional) Actualizar q_mem con flip-flop (aquí lo dejamos simple)
            # Ejemplo: S=1.0 si energy_val>5, R=1.0 si energy_val<2 (heurística).
            S_val = 1.0 if energy_val > 5 else 0.0
            R_val = 1.0 if energy_val < 2 else 0.0
            q_new_val = update_flip_flop(q_mem_anterior, S_val, R_val)
            
            # Clasificar emoción
            resultado = classify_emotion(pitch_val, energy_val, q_new_val)
            
            # Interpretación <5 => 'triste', >=5 => 'animada'
            if resultado < 5:
                etiqueta = "TRISTE"
            else:
                etiqueta = "ANIMADA"
            
            print(f"\nAudio: {fname}")
            print(f"  Pitch: {pitch_val:.2f} Hz, Energy: {energy_val:.2f}")
            print(f"  q_mem anter.: {q_mem_anterior:.2f} => new val: {q_new_val:.2f}")
            print(f"  => Emoción difusa: {resultado:.2f} => {etiqueta}")
            
            # Actualizar q_mem_anterior
            q_mem_anterior = q_new_val

###############################################################################
# EJECUCIÓN PRINCIPAL
###############################################################################

if __name__ == "__main__":
    audio_dir = "./audios_prueba"  # Ajusta según tu carpeta
    clasificar_audios(audio_dir)
