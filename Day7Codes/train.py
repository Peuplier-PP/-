from ultralytics import YOLO
import torch

if __name__ == '__main__':
    model = YOLO("yolov8n.yaml").load("yolov8n.pt")
    model.train(data="E:\实训\训练代码\Day7\dataset.yaml", imgsz=640, batch=16, epochs=20)