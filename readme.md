# YOLO11 Gloves Model

Este repositório contém um modelo treinado para detecção de luvas usando YOLOv11. O objetivo deste projeto é identificar e localizar mãos com ou sem luvas em imagens e vídeos com alta precisão.

## Características

* Baseado na arquitetura YOLOv11
* Treinado para detectar óculos em imagens
* Suporte para inferência em tempo real
* Código otimizado para GPU

## Estrutura do Repositório

```bash
yolo11_glasses_model/
│── runs/            
│── test/
│── train/
│── val/
│── data.yaml
│── camStream.py
│── trainingModel.py
│── README.md            
```

## Clone este repositório

```bash
git clone https://github.com/Baiokis/yolo11_gloves_model.git
cd yolo11_gloves_model
```

## Scripts

### 1. trainingModel

Este script realiza a função principal de treinar o modelo:

* Carrega uma configuração de modelo (yolo11n.yaml).
* Define hiperparâmetros como número de épocas, batch size, tamanho da imagem, otimizador (SGD), e outros.
* Inicia o treinamento usando os dados especificados em data.yaml.

### 2. camStream

Este script utiliza o modelo YOLO para realizar a detecção de luvas em tempo real usando a webcam.

* Carrega o modelo YOLO treinado para identificar luvas e ausência de luvas.
* Verifica se há GPU disponível e ajusta automaticamente para CUDA ou CPU.
* inicia comunicação com uma camera via IP.
* Desenha caixas ao redor dos objetos detectados, indicando a classe e confiança.
* Exibe o vídeo com as detecções em uma janela interativa.
