#!/usr/bin/env python3
import jetson.inference
import jetson.utils
import time
import csv
import pandas as pd

# -------------------------------------------------------
#  Load Pretrained Object Detection Network
# -------------------------------------------------------
net = jetson.inference.detectNet("ssd-mobilenet-v2", threshold=0.5)

# Setup camera and display
camera = jetson.utils.videoSource("/dev/video0")   # USB Camera
display = jetson.utils.videoOutput("display://0")  # On-screen window

# CSV setup
logfile = open("anomaly_log.csv", "w", newline="")
writer = csv.writer(logfile)
writer.writerow(["timestamp", "anomaly_reason", "detected_objects"])

# Initialize Pandas DataFrame
df = pd.DataFrame(columns=["timestamp", "anomaly_reason", "detected_objects"])

# -------------------------------------------------------
#  Detection Loop
# -------------------------------------------------------
frame_count = 0
save_interval = 10  # Save Pandas CSV every 10 anomaly frames

while display.IsStreaming():
    img = camera.Capture()
    detections = net.Detect(img)

    # Collect detected object labels
    classes = [net.GetClassDesc(d.ClassID).lower() for d in detections]
    anomaly_reason = None

    print("Detected:", classes)

    # ---------------------------------------------------
    # RULE 1: Food items
    # ---------------------------------------------------
    food_items = ["apple", "banana", "sandwich", "orange", "bottle", "cup"]
    for f in food_items:
        if f in classes:
            anomaly_reason = f"Food detected: {f}"
            break

    # ---------------------------------------------------
    # RULE 2: Mobile phones
    # ---------------------------------------------------
    if "cell phone" in classes or "mobile" in classes:
        anomaly_reason = "Mobile phone detected"
    
    # ---------------------------------------------------
    # RULE 3: Two or more people in frame
    # ---------------------------------------------------
    person_count = classes.count("person")
    if person_count >= 2:
        anomaly_reason = f"Multiple people detected: {person_count} persons"

    # ---------------------------------------------------
    # RULE 4: Empty chair detected
    # (if a 'chair' is seen but no 'person' in frame)
    # ---------------------------------------------------
    if "chair" in classes and person_count == 0:
        anomaly_reason = "Empty chair detected"

    # ---------------------------------------------------
    # Log and visualize anomalies
    # ---------------------------------------------------
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    if anomaly_reason:
        print(f"[ALERT] {timestamp} -> {anomaly_reason}")

        # Draw red rectangles for anomalies
        for d in detections:
            label = net.GetClassDesc(d.ClassID).lower()
            if (
                label in food_items or
                label in ["cell phone", "mobile", "chair", "person"]
            ):
                jetson.utils.cudaDrawRect(
                    img,
                    (int(d.Left), int(d.Top), int(d.Right), int(d.Bottom)),
                    (255, 0, 0)
                )

        # Write to CSV and pandas
        writer.writerow([timestamp, anomaly_reason, ", ".join(classes)])
        logfile.flush()
        df.loc[len(df)] = [timestamp, anomaly_reason, ", ".join(classes)]

        frame_count += 1
        if frame_count % save_interval == 0:
            df.to_csv("anomaly_log_pandas.csv", index=False)
            print(f"[INFO] Saved {len(df)} anomaly records to anomaly_log_pandas.csv")

        # Save anomaly screenshot
        filename = f"anomaly_{timestamp.replace(':', '-')}.jpg"
        jetson.utils.saveImage(filename, img)
        print(f"[INFO] Saved anomaly screenshot: {filename}")

    # Render video
    display.Render(img)
    display.SetStatus(f"Anomaly: {anomaly_reason if anomaly_reason else 'None'}")

# -------------------------------------------------------
#  Cleanup
# -------------------------------------------------------
if not df.empty:
    df.to_csv("anomaly_log_pandas.csv", index=False)
    print(f"[INFO] Final anomaly records saved: {len(df)}")
else:
    print("[INFO] No anomalies detected.")
logfile.close()
