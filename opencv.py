import cv2
import mediapipe as mp

# Inicializa módulos do MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Configuração da detecção de mãos
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Mapeamento de dedos para landmarks
FINGER_TIPS = [mp_hands.HandLandmark.INDEX_FINGER_TIP,
               mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
               mp_hands.HandLandmark.RING_FINGER_TIP,
               mp_hands.HandLandmark.PINKY_TIP,
               mp_hands.HandLandmark.THUMB_TIP]

FINGER_DIP = [mp_hands.HandLandmark.INDEX_FINGER_DIP,
              mp_hands.HandLandmark.MIDDLE_FINGER_DIP,
              mp_hands.HandLandmark.RING_FINGER_DIP,
              mp_hands.HandLandmark.PINKY_DIP,
              mp_hands.HandLandmark.THUMB_IP]


def count_fingers(hand_landmarks):
    """Conta dedos estendidos baseando-se na posição das juntas."""
    count = 0
    for tip, dip in zip(FINGER_TIPS, FINGER_DIP):
        # Para os 4 dedos (não polegar): compara eixo y
        if tip != mp_hands.HandLandmark.THUMB_TIP:
            if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[dip].y:
                count += 1
        else:
            # Para polegar: compara eixo x (polegar esquerdo/direito)
            if hand_landmarks.landmark[tip].x > hand_landmarks.landmark[dip].x:
                count += 1
    return count


def main():
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Erro ao acessar webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Converte BGR para RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Processa a imagem
        results = hands.process(image)

        # Desenha resultados
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        finger_count = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )
                finger_count = count_fingers(hand_landmarks)

        # Exibe texto na tela
        cv2.putText(
            image,
            f'Dedos: {finger_count}',
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2
        )

        cv2.imshow('Contagem de Dedos', image)
        if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
