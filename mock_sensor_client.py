import time
import requests
import random
import sys

# Target API URL
URL = "http://localhost:5000/api/sensor-data"

# States configuration (Mean voltage, standard deviation)
STATES = {
    "0": {"name": "Clean Water", "mean": 3.0, "std": 0.02},
    "1": {"name": "Low Contamination", "mean": 2.7, "std": 0.08},
    "2": {"name": "Medium Contamination", "mean": 2.1, "std": 0.20},
    "3": {"name": "High Contamination", "mean": 1.4, "std": 0.35}
}

def main():
    print("==================================================")
    print("  Universal Microplastic Detector - Mock Telemetry")
    print("==================================================")
    print("This client simulates real-time ESP32 voltage output.")
    print("Select a water contamination simulation mode:")
    print(" [0] - Clean Water (Baseline)")
    print(" [1] - Low Microplastic Concentration")
    print(" [2] - Medium Microplastic Concentration")
    print(" [3] - High Microplastic Concentration")
    print(" [q] - Quit")
    print("==================================================")

    current_state = "0"
    
    while True:
        # Prompt user to change mode (non-blocking simulation check)
        print(f"\n[Active Mode]: {STATES[current_state]['name']}")
        print("Press Enter to keep current, or enter [0-3] to switch state (q to quit): ", end="")
        
        # Read user input with default
        try:
            import msvcrt
            # Simple Windows key press check
            time_start = time.time()
            user_input = ""
            while time.time() - time_start < 2.0: # Check for input for 2 seconds
                if msvcrt.kbhit():
                    char = msvcrt.getwche()
                    if char in ["0", "1", "2", "3", "q", "Q"]:
                        user_input = char.lower()
                        break
                    elif char == "\r": # Enter pressed
                        break
            
            if user_input == "q":
                break
            elif user_input in STATES:
                current_state = user_input
                print(f"\n--> Switched to: {STATES[current_state]['name']}")
        except Exception:
            # Fallback if msvcrt isn't supported or errors
            # Read line directly (blocks for a short input or accepts stdin)
            try:
                line = sys.stdin.readline().strip()
                if line == "q":
                    break
                elif line in STATES:
                    current_state = line
            except KeyboardInterrupt:
                break

        # Generate 10 packets representing 2 seconds of telemetry (5Hz)
        cfg = STATES[current_state]
        print(f"Streaming 10 telemetry frames for: {cfg['name']}...")
        
        for _ in range(10):
            # Generate value with normal distribution + clip to 0-3.3V
            voltage = random.normalvariate(cfg["mean"], cfg["std"])
            voltage = max(0.0, min(3.3, voltage))
            
            payload = {"sensor_value": round(voltage, 4)}
            
            try:
                response = requests.post(URL, json=payload, timeout=1.0)
                if response.status_code == 200:
                    data = response.json()
                    status = data.get("prediction_text", "Buffering")
                    conf = data.get("confidence", 0.0)
                    mean = data.get("mean", 0.0)
                    std = data.get("std_dev", 0.0)
                    print(f"Sent: {voltage:.4f}V | Server ML Output: {status} (Confidence: {conf}% | Mean: {mean}V | StdDev: {std}V)")
                else:
                    print(f"Server returned error code: {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Connection error: Flask backend is not running at {URL}. Details: {e}")
                time.sleep(2)
                break
                
            time.sleep(0.2) # 5Hz transmission rate

if __name__ == "__main__":
    main()
