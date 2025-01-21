from ultralytics import YOLO
from pathlib import Path

def main():
    model = YOLO("yolo11n.yaml")

    HYPERPARAMETERS = {
        "epochs": 500,
        "batch_size": 32,
        "imgsz": 1080,
        "optimizer": "SGD",
        "device": "0",
        "workers": 8,
        "label_smoothing": 0.05,
    }

    results = model.train(
        data="data.yaml",
        epochs=HYPERPARAMETERS["epochs"],
        batch=HYPERPARAMETERS["batch_size"],
        imgsz=HYPERPARAMETERS["imgsz"],
        name=HYPERPARAMETERS["name"],
        optimizer=HYPERPARAMETERS["optimizer"],
        device=HYPERPARAMETERS["device"],
        workers=HYPERPARAMETERS["workers"],
        label_smoothing=HYPERPARAMETERS["label_smoothing"],
        pretrained=False
    )
    metrics = model.val()
    print(metrics)

#validar o modelo
def validate_images():#
    model = YOLO("modelos/best.pt")
    
    image_folder = Path("fotos")
    
    if not image_folder.exists() or not image_folder.is_dir():
        print(f"A pasta {image_folder} não existe ou não é um diretório válido.")
        return

    image_files = list(image_folder.glob("*.jpg")) + list(image_folder.glob("*.png"))
    
    if not image_files:
        print(f"Nenhuma imagem encontrada na pasta {image_folder}.")
        return

    for image_path in image_files:
        print(f"Validando imagem: {image_path}")
        results = model(image_path)
        
        for result in results:
            result.show()
            print(result)

if __name__ == "__main__":
    main()
