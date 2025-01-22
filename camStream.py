import cv2
import torch
from ultralytics import YOLO

# Configurações da câmera IP
username = "admin"
password = "autvix123456"
ip_camera = "192.168.1.108"
port = "554"
rtsp_url = f"rtsp://{username}:{password}@{ip_camera}:{port}/cam/realmonitor?channel=1&subtype=0"

# Inicializar o dispositivo
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Utilizando dispositivo: {device}")

# Carregar o modelo YOLOv8/11
model = YOLO('../modelos/best.pt')  # Certifique-se de que o modelo seja compatível

classNames = ['Gloves', 'no-gloves']

# Conectar à câmera
cap = cv2.VideoCapture(rtsp_url)

cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Erro ao conectar à câmera. Verifique as configurações.")
    exit()

print("Transmissão ao vivo iniciada com detecção de Oculos. Pressione 'q' para sair.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Não foi possível obter o frame da câmera.")
        break

    # Fazer inferência com o YOLOv8/11
    results = model.predict(source=frame, device=device, verbose=False)

    # Processar resultados
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])
        conf = round(float(result.conf[0]), 2)
        cls = int(result.cls[0])
        currentClass = classNames[cls]

        # Escolher cor do retângulo
        color = (0, 255, 0) if 'NO-' not in currentClass else (0, 0, 255)

        # Desenhar retângulo e rótulo
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{currentClass} {conf}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar o frame
    cv2.imshow("Transmissão ao Vivo com Detecção de Oculos", frame)

    # Sair ao pressionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
