#define SENSOR_PIN 34

void setup() {
  Serial.begin(115200);
}

void loop() {
  int raw = analogRead(SENSOR_PIN);
  float voltage = raw * (3.3 / 4095.0);
  Serial.printf("Raw: %d | Voltage: %.4f\n", raw, voltage);
  delay(200);
}