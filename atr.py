import cv2
import mediapipe as mp
import math

# Inicjalizacja kamery
cap = cv2.VideoCapture(0)

# Inicjalizacja MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Funkcja do obliczania kąta między dwiema prostymi
def calculate_angle(a, b, c):
    # Vektor AB (pierwsza prosta)
    ab = [b[0] - a[0], b[1] - a[1]]
    # Vektor BC (druga prosta)
    bc = [c[0] - b[0], c[1] - b[1]]
    
    # Iloczyn skalarny
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    
    # Moduł wektorów
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    
    # Kąt w radianach
    angle_radians = math.acos(dot_product / (magnitude_ab * magnitude_bc))
    
    # Zwrócenie kąta w stopniach
    return math.degrees(angle_radians)

# Ustawienie skalowalnego okna
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)  # Tworzy okno skalowalne
cv2.setWindowProperty("Video", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)  # Ustawienie trybu normalnego

# Przechwytywanie obrazu z kamery
while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Zmiana koloru obrazu na RGB (MediaPipe używa RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    # Jeśli wykryto postać
    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark
        
        # Współrzędne punktów
        foot_index = [landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].x, landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX].y]
        heel = [landmarks[mp_pose.PoseLandmark.LEFT_HEEL].x, landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y]
        ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y]
        
        # Zamiana współrzędnych z normalizowanych (0-1) na piksele
        h, w, _ = frame.shape
        foot_index = (int(foot_index[0] * w), int(foot_index[1] * h))
        heel = (int(heel[0] * w), int(heel[1] * h))
        ankle = (int(ankle[0] * w), int(ankle[1] * h))
        knee = (int(knee[0] * w), int(knee[1] * h))
        
        # Obliczanie kąta między dwiema prostymi
        angle = calculate_angle(heel, ankle, knee)
        
        # Przemiana kąta na "90 - kąt"
        adjusted_angle = 90 - angle
        
        # Rysowanie linii między punktami
        cv2.line(frame, foot_index, heel, (0, 255, 0), 3)  # Linia stopy (palec -> pięta)
        cv2.line(frame, ankle, knee, (0, 255, 0), 3)       # Linia nogi (kostka -> kolano)
        
        # Rysowanie punktów
        cv2.circle(frame, foot_index, 5, (0, 0, 255), -1)  # Punkt palca
        cv2.circle(frame, heel, 5, (0, 0, 255), -1)       # Punkt pięty
        cv2.circle(frame, ankle, 5, (0, 0, 255), -1)      # Punkt kostki
        cv2.circle(frame, knee, 5, (0, 0, 255), -1)       # Punkt kolana
        
        # Wyświetlanie kąta
        cv2.putText(frame, f"Angle: {int(adjusted_angle)}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Zachowanie proporcji obrazu
    window_width = 800  # szerokość okna
    window_height = int((frame.shape[0] / frame.shape[1]) * window_width)  # wysokość obliczona na podstawie proporcji
    resized_frame = cv2.resize(frame, (window_width, window_height))
    
    # Wyświetlanie obrazu
    cv2.imshow("Video", resized_frame)
    
    # Przerwanie w przypadku naciśnięcia klawisza 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Zakończenie działania programu
cap.release()
cv2.destroyAllWindows()
