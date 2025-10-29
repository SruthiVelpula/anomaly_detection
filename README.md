# Real-Time Anomaly Detection System (Jetson Nano + SSD-Mobilenet-V2)

This project implements a real-time anomaly detection system using NVIDIA Jetson Nano and the `jetson-inference` library.  
It continuously monitors a live camera feed, detects objects using a pretrained SSD-Mobilenet-V2 model,  
and flags anomalies such as food items, mobile phones, multiple people, or empty chairs.

## Demo & Downloads
You can download example images and logs from the latest release:  
[**First Demo Assets (v0.1.0)**](https://github.com/SruthiVelpula/anomaly_detection/releases/tag/v0.1.0)

**Includes:**
- `anomaly_images.zip` — sample anomaly frames  
- `anomaly_log.csv` / `anomaly_log_pandas.csv` — example outputs



## Objective

The objective of this project is to design a lightweight, GPU-accelerated system capable of identifying unusual or unsafe scenarios  
in real time using object detection and rule-based logic on an edge device.

This project demonstrates how computer vision and AI can be applied to practical monitoring use cases  
such as workplace safety, lab monitoring, and behavioral analytics.


## System Requirements

### Hardware
- NVIDIA Jetson Nano (or other Jetson device with CUDA and TensorRT)
- USB camera connected to `/dev/video0`

### Software
- JetPack 4.x or 5.x (includes CUDA, cuDNN, TensorRT)
- `Jetson-inference` and `jetson-utils`
- Python 3
- `pandas` (install using `pip3 install pandas`)


## Output Files
anomaly_log.csv - Real-time log of detected anomalies
anomaly_log_pandas.csv - pandas DataFrame export (every 10 anomaly frames)
anomaly_*.jpg - Screenshots of each anomaly frame

## Project Structure

anomaly-detection-docker/
│
├── app.py # Main detection script
├── README.md # Project documentation
├── requirements.txt # Python dependencies
└── .gitignore # Files excluded from GitHub (logs, images)

---

## How It Works

1. **Object Detection:**  
   The Jetson Nano runs a pre-trained SSD-Mobilenet-V2 model for real-time object detection using TensorRT.

2. **Rule-Based Anomaly Logic:**  
   The system evaluates each frame and applies rules such as:
   - Food items detected: apple, banana, sandwich, bottle, cup  
   - Mobile phone detected  
   - Multiple people detected (two or more)  
   - Empty chair detected (chair without person)

3. **Logging and Visualization:**  
   - Saves all anomalies to `anomaly_log.csv` and `anomaly_log_pandas.csv`  
   - Draws red rectangles around detected anomalies  
   - Saves screenshots of anomaly frames (for example, `anomaly_2025-10-28-16-32-15.jpg`)

---

## How to Run

### Step 1: Install Dependencies

sudo apt-get update
pip3 install pandas

# Output Files
File Name	                Description
anomaly_log.csv	          Real-time log of detected anomalies
anomaly_log_pandas.csv	    pandas DataFrame export (every 10 anomaly frames)
anomaly_*.jpg	             Screenshots of each anomaly frame

Detected: ['person', 'chair']
[ALERT] 2025-10-28 16:05:13 -> Empty chair detected
[INFO] Saved anomaly screenshot: anomaly_2025-10-28-16-05-13.jpg
[INFO] Saved 10 anomaly records to anomaly_log_pandas.csv




## Use Cases
Workplace monitoring to detect unsafe conditions or absence of personnel
Cafeteria or laboratory analysis to identify the presence of food items or mobile phones in restricted areas
AI research or demonstration projects for real-time computer vision applications

## Future Enhancements
Integrate helmet or mask detection
Stream live alerts to a dashboard or web application
 

## Author
Sruthi Velpula
Graduate Student, MPS Data Science – University of Maryland, Baltimore County (UMBC)
LinkedIn: https://www.linkedin.com/in/sruthi-velpula/
GitHub: https://github.com/SruthiVelpula


## License

This project is open-source under the MIT License.  
You may use, modify, and share it for research or educational purposes.  
Please provide proper attribution to the author when referencing or reusing this work.
