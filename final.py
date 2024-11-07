from ultralytics import YOLO
import numpy as np
from collections import Counter

# Define PCU values for different vehicle types
PCU_VALUES = {
    'car': 1.0,
    'motorbike': 0.5,
    'bus': 2.5
}

# Define minimum and maximum green light durations (in seconds)
T_MIN = 10  # minimum green light time
T_MAX = 60  # maximum green light time

# Path to the model and the images for each signal
model = YOLO(r"C:\Users\vedant raikar\Desktop\yolov5\best.pt")
image_paths = {
    'signal_1': r"C:\Users\vedant raikar\Desktop\yolov5\download.jpg",
    'signal_2': r"C:\Users\vedant raikar\Desktop\yolov5\download.jpg",
    'signal_3': r"C:\Users\vedant raikar\Desktop\yolov5\test.jpg",
    'signal_4': r"C:\Users\vedant raikar\Desktop\yolov5\test.jpg"
}

def calculate_pcu(vehicles):
    """Calculate the total PCU for a given set of vehicles."""
    return sum(count * PCU_VALUES.get(vehicle_type, 0) for vehicle_type, count in vehicles.items())

def calculate_green_time(total_pcu, signal_pcu):
    """Calculate the green light duration for a signal based on its PCU proportion."""
    return T_MIN + (signal_pcu / total_pcu) * (T_MAX - T_MIN) if total_pcu > 0 else T_MIN

def calculate_waiting_time(signal_pcu, average_wait_time_per_vehicle=2):
    """Estimate the waiting time for a signal based on its total PCU."""
    return signal_pcu * average_wait_time_per_vehicle

def get_vehicle_counts(image_path):
    """Detect vehicles in an image and return their counts by category."""
    threshold = 0.001
    results = model.predict(source=image_path, imgsz=448, conf=threshold)
    
    # Count detected objects
    cls = results[0].boxes.cls.tolist()
    class_names = model.names  # Class names list
    class_labels = [class_names[int(label)] for label in cls]
    category_counts = Counter(class_labels)
    
    # Filter out non-vehicle classes (if needed)
    vehicle_counts = {k: category_counts.get(k, 0) for k in PCU_VALUES.keys()}
    return vehicle_counts

# Process each signal's image and calculate green time and waiting time
signal_data = {}
for signal, image_path in image_paths.items():
    vehicle_counts = get_vehicle_counts(image_path)
    signal_pcu = calculate_pcu(vehicle_counts)
    signal_data[signal] = {
        'vehicle_counts': vehicle_counts,
        'pcu': signal_pcu
    }

# Calculate total PCU across all signals
total_pcu = sum(data['pcu'] for data in signal_data.values())

# Calculate green light duration and waiting time for each signal
for signal, data in signal_data.items():
    green_time = calculate_green_time(total_pcu, data['pcu'])
    waiting_time = calculate_waiting_time(data['pcu'])
    
    print(f"{signal}:\n - Green Light Duration: {green_time:.2f} seconds\n - Waiting Time: {waiting_time:.2f} seconds\n")
