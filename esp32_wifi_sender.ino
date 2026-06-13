#include <WiFi.h>
#include <HTTPClient.h>

// ==========================================
//   Wi-Fi & Server Configurations
// ==========================================
const char* ssid = "wifi ssid";          // Replace with your Wi-Fi SSID
const char* password = "password";  // Replace with your Wi-Fi Password

// Flask Backend API URL (Replace with your computer's local network IP)
const char* serverUrl = "http://my ip:5000/api/sensor-data"; 

// ==========================================
//   Hardware Pin Configuration
// ==========================================
#define SENSOR_PIN 34   // GPIO Pin connected to photodiode output
#define LASER_PIN 2     // GPIO Pin connected to laser diode (and onboard LED)

void setup() {
  Serial.begin(115200);
  pinMode(LASER_PIN, OUTPUT);
  
  // Connect to Wi-Fi
  Serial.println();
  Serial.print("Connecting to Wi-Fi: ");
  Serial.println(ssid);
  
  WiFi.begin(ssid, password);
  
  // Blink laser/LED during WiFi connection phase
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(LASER_PIN, HIGH);
    delay(250);
    digitalWrite(LASER_PIN, LOW);
    delay(250);
    Serial.print(".");
  }
  
  Serial.println("");
  Serial.println("WiFi connected successfully!");
  Serial.print("ESP32 IP Address: ");
  Serial.println(WiFi.localIP());
  
  // Laser stays ON constantly during active operation for DLS scanning
  digitalWrite(LASER_PIN, HIGH);
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    // Read raw ADC value (12-bit resolution: 0 - 4095)
    int rawValue = analogRead(SENSOR_PIN);
    
    // Convert ADC to raw voltage (Assuming 3.3V reference)
    float voltage = (rawValue * 3.3) / 4095.0;
    
    // Create JSON Payload
    String jsonPayload = "{\"sensor_value\":" + String(voltage, 4) + "}";
    
    // Send HTTP POST request
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");
    
    Serial.print("Sending voltage telemetry: ");
    Serial.print(voltage);
    Serial.println(" V");
    
    int httpResponseCode = http.POST(jsonPayload);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.print("Server HTTP Status: ");
      Serial.println(httpResponseCode);
      Serial.print("Server Response: ");
      Serial.println(response);
    } else {
      Serial.print("Error sending POST request: ");
      Serial.println(httpResponseCode);
    }
    
    http.end();
  } else {
    Serial.println("WiFi disconnected! Reconnecting...");
    WiFi.begin(ssid, password);
  }
  
  // Wait 200ms (5Hz telemetry frequency)
  delay(200);
}
