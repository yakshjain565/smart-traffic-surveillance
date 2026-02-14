"""
PRO VEHICLE DETECTION & COUNTING SYSTEM
Minimal • Modern • CV-Ready
Tkinter + OpenCV + YOLOv8 Deep Learning

Author: Yaksh Jain (Editable)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import csv
import datetime
import threading
import os

# ---------------- CONFIG ----------------
WINDOW_TITLE = "Smart Traffic Surveillance System"
DETECTION_LINE_Y = 320
CONFIDENCE_THRESHOLD = 0.4

# Load YOLOv8 Model
try:
    from ultralytics import YOLO
    MODEL = YOLO("yolov8n.pt")
except:
    MODEL = None
    print("YOLO not loaded. Install ultralytics.")

# YOLO class uses "motorcycle" not "motorbike"
VEHICLE_CLASSES = ["car", "bus", "truck", "motorcycle"]

# Class Colors
CLASS_COLORS = {
    "car": (0, 255, 0),
    "bus": (255, 165, 0),
    "truck": (255, 0, 0),
    "motorcycle": (0, 200, 255)
}

# ---------------- APP CLASS ----------------
class VehicleCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.root.geometry("1150x720")
        self.root.configure(bg="#121212")

        self.video_path = None
        self.cap = None
        self.running = False

        self.total_count = 0
        self.up_count = 0
        self.down_count = 0
        self.tracked_objects = {}
        self.object_id = 0

        self.create_ui()

    # ---------------- UI ----------------
    def create_ui(self):
        header = tk.Frame(self.root, bg="#1f1f1f", height=60)
        header.pack(fill=tk.X)

        title = tk.Label(header, text="Smart Traffic Surveillance System",
                         font=("Segoe UI", 20, "bold"),
                         fg="white", bg="#1f1f1f")
        title.pack(pady=10)

        # Stats Bar
        stats_bar = tk.Frame(self.root, bg="#181818")
        stats_bar.pack(fill=tk.X, pady=5)

        self.total_label = tk.Label(stats_bar, text="TOTAL: 0",
                                    font=("Segoe UI", 14, "bold"),
                                    fg="#00ff99", bg="#181818")
        self.total_label.pack(side=tk.LEFT, padx=20)

        self.up_label = tk.Label(stats_bar, text="UP: 0",
                                 font=("Segoe UI", 14, "bold"),
                                 fg="#00ccff", bg="#181818")
        self.up_label.pack(side=tk.LEFT, padx=20)

        self.down_label = tk.Label(stats_bar, text="DOWN: 0",
                                   font=("Segoe UI", 14, "bold"),
                                   fg="#ff6666", bg="#181818")
        self.down_label.pack(side=tk.LEFT, padx=20)

        # Video Display
        self.video_label = tk.Label(self.root, bg="#121212")
        self.video_label.pack(pady=10)

        # Controls
        controls = tk.Frame(self.root, bg="#121212")
        controls.pack(pady=10)

        self.make_button(controls, "Open Video", self.load_video, 0)
        self.make_button(controls, "Start", self.start, 1)
        self.make_button(controls, "Stop", self.stop, 2)
        self.make_button(controls, "Snapshot", self.snapshot, 3)
        self.make_button(controls, "Export CSV", self.export_csv, 4)

        # Logs
        self.log_box = tk.Text(self.root, height=6, bg="#0a0a0a", fg="#00ff99",
                               font=("Consolas", 10))
        self.log_box.pack(fill=tk.X, padx=15, pady=10)

        self.log("System Initialized")

    def make_button(self, parent, text, cmd, col):
        btn = tk.Button(parent, text=text, command=cmd,
                        bg="#1f1f1f", fg="white",
                        activebackground="#00ff99",
                        activeforeground="black",
                        font=("Segoe UI", 11, "bold"),
                        width=14, bd=0)
        btn.grid(row=0, column=col, padx=8)

    # ---------------- CORE ----------------
    def load_video(self):
        path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
        if path:
            self.video_path = path
            self.log(f"Loaded: {os.path.basename(path)}")

    def start(self):
        if not self.video_path:
            messagebox.showwarning("Warning", "Please load a video first")
            return

        self.cap = cv2.VideoCapture(self.video_path)
        self.running = True
        threading.Thread(target=self.run_detection, daemon=True).start()
        self.log("Detection Started")

    def stop(self):
        self.running = False
        self.log("Detection Stopped")

    # ---------------- DETECTION ----------------
    def run_detection(self):
        if MODEL is None:
            messagebox.showerror("Error", "YOLO model not loaded")
            return

        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (900, 500))
            results = MODEL(frame)[0]

            cv2.line(frame, (0, DETECTION_LINE_Y),
                     (900, DETECTION_LINE_Y), (0, 255, 255), 2)

            for box in results.boxes:
                cls_id = int(box.cls[0])
                label = MODEL.names[cls_id]

                if label in VEHICLE_CLASSES and float(box.conf[0]) > CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)

                    obj_id = self.track_object(cx, cy)
                    color = CLASS_COLORS[label]

                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.circle(frame, (cx, cy), 4, color, -1)

                    cv2.putText(frame,
                                f"{label.upper()} | ID:{obj_id}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)

            self.update_stats()
            self.display_frame(frame)

        self.cap.release()

    # ---------------- TRACKING ----------------
    def track_object(self, cx, cy):
        assigned_id = None

        for obj_id, (px, py) in self.tracked_objects.items():
            if abs(cx - px) < 40 and abs(cy - py) < 40:
                assigned_id = obj_id
                break

        if assigned_id is None:
            assigned_id = self.object_id
            self.tracked_objects[self.object_id] = (cx, cy)
            self.object_id += 1
            return assigned_id

        px, py = self.tracked_objects[assigned_id]

        if py < DETECTION_LINE_Y and cy > DETECTION_LINE_Y:
            self.down_count += 1
            self.total_count += 1
            self.log(f"Vehicle {assigned_id} → DOWN")

        elif py > DETECTION_LINE_Y and cy < DETECTION_LINE_Y:
            self.up_count += 1
            self.total_count += 1
            self.log(f"Vehicle {assigned_id} → UP")

        self.tracked_objects[assigned_id] = (cx, cy)
        return assigned_id

    # ---------------- DISPLAY ----------------
    def display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(img)

        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    # ---------------- TOOLS ----------------
    def snapshot(self):
        if hasattr(self.video_label, "imgtk"):
            filename = f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            self.video_label.imgtk._PhotoImage__photo.write(filename)
            self.log(f"Snapshot saved: {filename}")

    def export_csv(self):
        filename = f"traffic_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Total Vehicles", "Up", "Down"])
            writer.writerow([
                datetime.datetime.now(),
                self.total_count,
                self.up_count,
                self.down_count
            ])

        self.log(f"CSV Exported: {filename}")

    # ---------------- LOGGING ----------------
    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert(tk.END, f"[{timestamp}] {msg}\n")
        self.log_box.see(tk.END)

    def update_stats(self):
        self.total_label.config(text=f"TOTAL: {self.total_count}")
        self.up_label.config(text=f"UP: {self.up_count}")
        self.down_label.config(text=f"DOWN: {self.down_count}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    root = tk.Tk()
    app = VehicleCounterApp(root)
    root.mainloop()

"""
---------------- CV DESCRIPTION ----------------
Smart Traffic Surveillance System
• Built a deep learning-based traffic monitoring system using YOLOv8 and OpenCV
• Designed a modern dashboard-style GUI using Tkinter
• Implemented real-time multi-class vehicle detection including two-wheelers
• Developed object tracking with directional traffic flow analysis
• Added CSV reporting, snapshot capture, and live system logs

Tech Stack: Python, OpenCV, Tkinter, YOLOv8, NumPy, Pillow
----------------------------------------------
"""