import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def generate_synthetic_data(num_samples_per_class=100, window_size=10):
    """
    Generates synthetic voltage readings representing clean and contaminated water.
    Under Dynamic Light Scattering (DLS):
    - Clean: High voltage (clear light path), low noise (low std dev)
    - Low Contamination: Slightly lower voltage, moderate noise
    - Medium Contamination: Lower average voltage, high noise
    - High Contamination: Very low voltage, extremely high noise
    """
    np.random.seed(42)
    data = []

    # Target class definitions:
    # 0 = Clean, 1 = Low Contamination, 2 = Medium Contamination, 3 = High Contamination
    configs = {
        0: {"mean": 3.0, "std": 0.02}, # Clean
        1: {"mean": 2.7, "std": 0.08}, # Low
        2: {"mean": 2.1, "std": 0.20}, # Medium
        3: {"mean": 1.4, "std": 0.35}  # High
    }

    for label, config in configs.items():
        for _ in range(num_samples_per_class):
            # Generate raw voltages for a window
            raw_voltages = np.random.normal(config["mean"], config["std"], window_size)
            # Clip between 0V and 3.3V (ESP32 ADC voltage limits)
            raw_voltages = np.clip(raw_voltages, 0.0, 3.3)
            
            # Extract features
            mean_val = np.mean(raw_voltages)
            std_dev = np.std(raw_voltages)
            min_val = np.min(raw_voltages)
            max_val = np.max(raw_voltages)
            range_val = max_val - min_val

            data.append({
                'mean_val': mean_val,
                'std_dev': std_dev,
                'min_val': min_val,
                'max_val': max_val,
                'range_val': range_val,
                'label': label
            })

    df = pd.DataFrame(data)
    return df

def main():
    print("Generating synthetic microplastic sensor calibration data...")
    df = generate_synthetic_data(num_samples_per_class=150)
    
    # Save training dataset to CSV
    os.makedirs("D:\\Microplastic-Detection-System\\backend", exist_ok=True)
    csv_path = "D:\\Microplastic-Detection-System\\backend\\microplastic_training_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"Dataset saved to {csv_path}")

    # Prepare features and labels
    X = df[['mean_val', 'std_dev', 'min_val', 'max_val', 'range_val']]
    y = df['label']

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest Classifier
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Clean", "Low", "Medium", "High"]))

    # Save the trained model
    model_path = "D:\\Microplastic-Detection-System\\backend\\microplastic_model.pkl"
    joblib.dump(model, model_path)
    print(f"\nTrained model successfully saved to: {model_path}")

if __name__ == "__main__":
    main()
