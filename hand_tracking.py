import cv2
import mediapipe as mp
import numpy as np
import math

# --- Configuración Visual ---
COLOR_CIAN = (255, 255, 0) # BGR
COLOR_MAGENTA = (255, 0, 255)
COLOR_BLANCO = (255, 255, 255)
ESPESOR_LINEA = 2
RADIO_NODO = 3

# --- Configuración de la Cuadrícula 3D ---
TAMAÑO_GRID = 60 # Tamaño de cada celda del cubo
GRID_DEPTH_MAX = 5 # Cuántas capas de profundidad queremos simular

# --- Inicialización de MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_drawing = mp.solutions.drawing_utils

# --- Estado del Mundo 3D ---
# Lista de tuplas (gx, gy, gz) que representan las coordenadas de la cuadrícula de cada cubo colocado
cubos_colocados = []
# Coordenadas de la mano en el frame anterior para suavizar movimiento
smoothed_hand_pos = np.array([0, 0, 0], dtype=np.float32)
frame_count = 0

# --- Funciones de Renderizado Isométrico ---

def proyectar_a_isometrico(gx, gy, gz, offset_x=0, offset_y=0):
    """
    Convierte coordenadas de cuadrícula 3D (gx, gy, gz) a coordenadas de pantalla 2D (px, py)
    usando una proyección isométrica simple.
    """
    # Ecuaciones estándar de proyección isométrica
    # px = (gx - gz) * cos(30°)
    # py = (gx + gz) * sin(30°) + gy
    
    ang_x = math.radians(30)
    ang_z = math.radians(30)
    
    # Factor de escala visual
    sc = TAMAÑO_GRID / 2.0
    
    px = (gx * math.cos(ang_x) - gz * math.cos(ang_z)) * sc
    py = (gx * math.sin(ang_x) + gz * math.sin(ang_z)) * sc + gy * sc
    
    # Invertimos Y para que gy positivo vaya hacia arriba
    py = -py 
    
    return (int(px + offset_x), int(py + offset_y))

def dibujar_cubo_isometrico(img, center_screen_x, center_screen_y, tamano, line_color, node_color=None):
    """Dibuja un wireframe de un cubo isométrico centrado en un punto de pantalla."""
    s = tamano / 2.0
    d = tamano * 0.4 # Offset visual para la "profundidad" isométrica (offset de los vértices traseros)
    
    # Definir los 8 vértices en 2D que forman la proyección isométrica
    # (Esto es una aproximación visual basada en los 8 vértices de un cubo real proyectados)
    v = np.array([
        [-s, -s/2.0],        # V0: Delantero superior izquierdo
        [s, -s/2.0],         # V1: Delantero superior derecho
        [s, s/2.0],          # V2: Delantero inferior derecho
        [-s, s/2.0],         # V3: Delantero inferior izquierdo
        [-s + d, -s/2.0 - d], # V4: Trasero superior izquierdo
        [s + d, -s/2.0 - d],  # V5: Trasero superior derecho
        [s + d, s/2.0 - d],   # V6: Trasero inferior derecho
        [-s + d, s/2.0 - d]   # V7: Trasero inferior izquierdo
    ])
    
    # Definir las 12 aristas (pares de índices de vértices a conectar)
    aristas = [
        (0, 1), (1, 2), (2, 3), (3, 0), # Cara frontal
        (4, 5), (5, 6), (6, 7), (7, 4), # Cara trasera
        (0, 4), (1, 5), (2, 6), (3, 7)  # Conexiones frontales-traseras
    ]
    
    # Sumar el centro de la pantalla
    v_screen = v + np.array([center_screen_x, center_screen_y])
    v_ints = v_screen.astype(int)
    
    # Dibujar las líneas
    for p1_idx, p2_idx in aristas:
        cv2.line(img, tuple(v_ints[p1_idx]), tuple(v_ints[p2_idx]), line_color, ESPESOR_LINEA)
        
    # Dibujar los nodos brillantes en los vértices
    if node_color:
        for vert in v_ints:
            cv2.circle(img, tuple(vert), RADIO_NODO, node_color, -1)

# --- Bucle Principal ---

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Invertir el fotograma horizontalmente para que actúe como un espejo
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    # Definir el centro del mundo 3D en la pantalla
    # Lo colocamos un poco más abajo para que la cuadrícula se extienda hacia arriba
    center_world_x = w // 2
    center_world_y = h // 2 + 100

    # Convertir a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    pos_dedo = None
    distancia_dedos = 1000 # Inicializar con valor alto

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Obtener puntos clave
            landmark_8 = hand_landmarks.landmark[8] # Punta índice
            landmark_4 = hand_landmarks.landmark[4] # Punta pulgar

            # Convertir a píxeles
            pos_indice = (int(landmark_8.x * w), int(landmark_8.y * h))
            pos_pulgar = (int(landmark_4.x * w), int(landmark_4.y * h))
            
            # Usar el índice como "punto de mira"
            pos_dedo = pos_indice

            # Calcular distancia entre dedos (para el gesto de "click")
            distancia_dedos = math.hypot(pos_indice[0] - pos_pulgar[0], pos_indice[1] - pos_pulgar[1])
            
            # Simular profundidad (z) basada en el tamaño de la mano
            # A mano más grande (más cerca), z es menor.
            # Calculamos la distancia 3D aproximada entre pulgar y muñeca (Landmark 0)
            wrist = hand_landmarks.landmark[0]
            hand_size_2d = math.hypot(landmark_4.x - wrist.x, landmark_4.y - wrist.y)
            # Normalizar tamaño visual a coordenadas de cuadrícula z (0 a GRID_DEPTH_MAX)
            # Estos valores (-1.0, 0.4) pueden necesitar ajuste según tu cámara
            z_grid = int(np.interp(hand_size_2d, [-1.0, 0.4], [GRID_DEPTH_MAX, 0]))
            z_grid = np.clip(z_grid, 0, GRID_DEPTH_MAX)

            # Dibujar la mano de MediaPipe (opcional, ayuda a depurar)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # --- Renderizado del Mundo 3D ---

    # Dibujar primero todos los cubos ya colocados
    for cx, cy, cz in cubos_colocados:
        px, py = proyectar_a_isometrico(cx, cy, cz, center_world_x, center_world_y)
        # Dibujar cubo sólido cian con nodos magentas
        dibujar_cubo_isometrico(frame, px, py, TAMAÑO_GRID, COLOR_CIAN, COLOR_MAGENTA)

    # Si se detecta un dedo, dibujar el "cubo fantasma" (el que estás a punto de colocar)
    if pos_dedo:
        # 1. Convertir la posición 2D de la pantalla a coordenadas de cuadrícula (gx, gy)
        # Esto es lo opuesto a la proyección isométrica, es complejo pero podemos aproximarlo
        # porque conocemos z_grid.
        # Aproximación simple:
        ang_x = math.radians(30)
        ang_z = math.radians(30)
        sc = TAMAÑO_GRID / 2.0
        
        # Deshacer el offset del centro del mundo
        raw_px = pos_dedo[0] - center_world_x
        raw_py = -(pos_dedo[1] - center_world_y) # Invertir Y
        
        # Ecuaciones inversas aproximadas (con z conocido)
        # gy * sc = raw_py - z_grid * sc * sin(30) - gx * sc * sin(30) -- esto es difícil
        
        # Una forma más intuitiva es calcular gx, gy de forma más simple y dejar que z simule profundidad:
        # (Esto es lo que permite que el cubo siga el dedo y se una a la red)
        
        # Calcular gx, gy de cuadrícula cruda basadas en la pantalla
        gx_raw = raw_px / (sc * math.cos(ang_x))
        gy_raw = (raw_py - z_grid * sc * math.sin(ang_z)) / sc
        
        # Redondear gx_raw, gy_raw para "ajustarse" a la celda más cercana de la cuadrícula
        gx_grid = round(gx_raw)
        gy_grid = round(gy_raw)
        
        # 2. Volver a proyectar las coordenadas de la cuadrícula ajustada para dibujar el "fantasma"
        px_snapped, py_snapped = proyectar_a_isometrico(gx_grid, gy_grid, z_grid, center_world_x, center_world_y)
        
        # Dibujar un cubo "fantasma" (líneas blancas más finas, sin nodos magentas)
        dibujar_cubo_isometrico(frame, px_snapped, py_snapped, TAMAÑO_GRID, COLOR_BLANCO)
        
        # Dibujar un "punto de mira" en la posición de redonda (magenta brillante)
        cv2.circle(frame, (px_snapped, py_snapped), 5, COLOR_MAGENTA, -1)

        # --- Detección de Colocación (Click) ---
        # Si la distancia entre dedos es pequeña (menos de 25 píxeles), hacemos "click"
        if distancia_dedos < 25:
            # Solo agregamos si no hay un cubo ya en esa posición
            pos_nueva = (gx_grid, gy_grid, z_grid)
            if pos_nueva not in cubos_colocados:
                cubos_colocados.append(pos_nueva)
                # Opcional: imprimir en terminal para confirmar
                print(f"Cubo colocado en cuadrícula: {pos_nueva}")

    # --- UI Elements ---
    cv2.putText(frame, f"Cubos: {len(cubos_colocados)}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, COLOR_BLANCO, 2)
    cv2.putText(frame, "Pellizca para colocar, 'C' para borrar, 'Q' para salir", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_CIAN, 1)

    # Mostrar el fotograma final con superposición
    cv2.imshow("Mundo de Vóxeles de Mano (Isometrico)", frame)

    # Teclas de control
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('c'): cubos_colocados = [] # Borrar todo

cap.release()
cv2.destroyAllWindows()