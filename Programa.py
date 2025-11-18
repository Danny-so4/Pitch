#!/usr/bin/env python3
# Main2.py - versión con salida extendida de confianzas

import sys
import time
import logging
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import pigpio
import RPi.GPIO as GPIO
from picamera2 import Picamera2

# ==== CONFIG LOGGING ====
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")

# ==== CONFIGURACIÓN ====
SERVO1, SERVO2 = 12, 13
S1_MIN, S1_MAX, S1_CENTER = 2, 178, 88
S2_MIN, S2_MAX, S2_CENTER = 0, 170, 85

TRIG, ECHO = 27, 17
DIST_MIN_CM, DIST_MAX_CM = 3, 26
CONSISTENCIA_TOL = 5.0

MOVE_CENTER_DELAY = 2.0

MODEL_PATH = "converted_tflite/model_unquant.tflite"
FRAME_SIZE = (640, 480)
INPUT_SIZE = (224, 224)
LABELS = ["Orgánicos", "Aprovechables", "No aprovechables", "Peligrosos"]
POSITIONS = {
    0: (138, 0),
    1: (33, 170),
    2: (33, 0),
    3: (138, 170)
}

# ==== FUNCIONES BÁSICAS ====

def mover_servo(pin, angulo, ang_min, ang_max):
    angulo = max(ang_min, min(angulo, ang_max))
    pulso = int((angulo / 180.0) * 2000 + 500)
    pi.set_servo_pulsewidth(pin, pulso)

def mover_a_posicion(pos):
    ang1, ang2 = pos
    mover_servo(SERVO1, ang1, S1_MIN, S1_MAX)
    mover_servo(SERVO2, ang2, S2_MIN, S2_MAX)
    logging.info(f"Servos movidos a posición: {pos}")

def volver_centro():
    mover_servo(SERVO1, S1_CENTER, S1_MIN, S1_MAX)
    mover_servo(SERVO2, S2_CENTER, S2_MIN, S2_MAX)
    logging.info("Servos en posición central")

# ==== SENSOR ULTRASÓNICO ====
def medir_distancia():
    GPIO.output(TRIG, False)
    time.sleep(0.05)

    GPIO.output(TRIG, True)
    time.sleep(0.00001)
    GPIO.output(TRIG, False)

    timeout = time.monotonic() + 0.04
    while GPIO.input(ECHO) == 0 and time.monotonic() < timeout:
        pass
    if GPIO.input(ECHO) == 0:
        return None
    inicio = time.monotonic()

    timeout = time.monotonic() + 0.04
    while GPIO.input(ECHO) == 1 and time.monotonic() < timeout:
        pass
    if GPIO.input(ECHO) == 1:
        return None
    fin = time.monotonic()

    duracion = fin - inicio
    dist = (duracion * 34300) / 2
    if DIST_MIN_CM <= dist <= DIST_MAX_CM:
        return round(dist, 2)
    return None

def lecturas_consistentes(n=3, tolerancia=CONSISTENCIA_TOL):
    medidas = []
    for _ in range(n):
        d = medir_distancia()
        if d is None:
            return None
        medidas.append(d)
        time.sleep(0.05)
    logging.info(f"Lecturas: [{', '.join(f'{m:.2f}cm' for m in medidas)}]")
    if max(medidas) - min(medidas) <= tolerancia:
        return round(sum(medidas) / len(medidas), 2)
    return None

# ==== MODELO ====
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def preparar_input(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, INPUT_SIZE)
    arr = img.astype(np.float32) / 127.5 - 1.0
    arr = np.expand_dims(arr, 0)
    return arr

def inferir(frame):
    x = preparar_input(frame)
    interpreter.set_tensor(input_details[0]['index'], x)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    probs = softmax(out.flatten())
    clase = int(np.argmax(probs))
    return probs, clase

def mover_por_clase(clase):
    if clase in POSITIONS:
        a1, a2 = POSITIONS[clase]
        mover_servo(SERVO1, a1, S1_MIN, S1_MAX)
        mover_servo(SERVO2, a2, S2_MIN, S2_MAX)
        logging.info(f"Moviendo servos para clase: {LABELS[clase]}")
        time.sleep(MOVE_CENTER_DELAY)
        volver_centro()

# ==== INICIALIZACIÓN ====
pi = pigpio.pi()
if not pi.connected:
    logging.error("pigpiod no está corriendo. Saliendo.")
    sys.exit(1)

GPIO.setmode(GPIO.BCM)
GPIO.setup(TRIG, GPIO.OUT)
GPIO.setup(ECHO, GPIO.IN)
GPIO.output(TRIG, False)

try:
    picam2 = Picamera2()
    cfg = picam2.create_preview_configuration(main={"size": FRAME_SIZE})
    picam2.configure(cfg)
    picam2.start()
except Exception as e:
    logging.error(f"Error al iniciar cámara: {e}")
    pi.stop()
    GPIO.cleanup()
    sys.exit(1)

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ==== LOOP PRINCIPAL ====
def main_loop():
    try:
        logging.info("Sistema iniciado. Centrando servos...")
        volver_centro()
        time.sleep(1)

        while True:
            dist = medir_distancia()
            logging.info(f"Distancia medida: {dist} cm")
            if dist is None:
                time.sleep(0.1)
                continue

            confirmada = lecturas_consistentes()
            if confirmada is None:
                logging.info("Lecturas inconsistentes o inválidas.")
                continue

            logging.info(f"Distancia promedio confirmada: {confirmada} cm")
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            probs, clase = inferir(frame)
            confianza = probs[clase] * 100

            # Mostrar todas las confianzas
            msg_conf = " | ".join(f"{LABELS[i]}: {probs[i]*100:.2f}%" for i in range(len(LABELS)))
            logging.info(f"Confianzas → {msg_conf}")
            logging.info(f"Clase seleccionada: {LABELS[clase]} ({confianza:.2f}%)")

            mover_por_clase(clase)
            time.sleep(0.5)

    except KeyboardInterrupt:
        logging.info("Interrupción del usuario (Ctrl+C).")
    finally:
        picam2.stop()
        pi.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        logging.info("Recursos liberados correctamente. Fin del programa.")

if __name__ == "__main__":
    main_loop()
