// ===================== INCLUDES =====================
#include <Wire.h>
#include <BH1750.h>
#include <Adafruit_NeoPixel.h>

// ===================== BH1750 & NEOPIXEL =====================
BH1750 lightMeter;

#define LED_PIN 18
#define NUM_LEDS 30

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

int currentColor = -1;

// ===================== PIN DEFINITIONS =====================
// Float Sensor
#define FLOAT_PIN     27
#define RELAY_FLOAT   26

// MQ135 Sensor
#define MQ135_PIN     34
#define RELAY_MQ135   25

// TDS Sensor
#define TDS_PIN       32
#define RELAY_TDS     15

// pH Sensor
#define PH_PIN        35
#define RELAY_PH      33

// ===================== THRESHOLDS =====================
float MQ_THRESHOLD = 313.5;
const int TDS_THRESHOLD = 80;

// ===================== pH CALIBRATION =====================
float voltage_acid = 1.261;
float ph_acid = 3.50;
float voltage_neutral = 1.115;
float ph_neutral = 7.00;
float voltage_alkaline = 0.142;
float ph_alkaline = 8.00;

// ===================== FUNCTIONS =====================

// ---- NeoPixel Function ----
void setAll(uint8_t r, uint8_t g, uint8_t b) {
  strip.clear();

  for (int i = 0; i < NUM_LEDS; i++) {
    strip.setPixelColor(i, strip.Color(r, g, b));
  }

  strip.show();
}

// ---- Median Filter for pH ----
float readMedian(int pin, int samples) {
  int arr[25];

  for (int i = 0; i < samples; i++) {
    arr[i] = analogRead(pin);
    delay(5);
  }


  for (int i = 0; i < samples - 1; i++) {
    for (int j = i + 1; j < samples; j++) {
      if (arr[j] < arr[i]) {
        int temp = arr[i];
        arr[i] = arr[j];
        arr[j] = temp;
      }
    }
  }

  return arr[samples / 2];
}

// ---- Calculate pH ----
float readPH() {
  float adc = readMedian(PH_PIN, 25);
  float voltage = adc * 3.3 / 4095.0;
  float ph;

  if (voltage >= voltage_neutral) {
    ph = ph_neutral +
         (ph_acid - ph_neutral) *
         (voltage - voltage_neutral) /
         (voltage_acid - voltage_neutral);
  } else {
    ph = ph_neutral +
         (ph_alkaline - ph_neutral) *
         (voltage - voltage_neutral) /
         (voltage_alkaline - voltage_neutral);
  }

  if (ph < 0) ph = 0;
  if (ph > 14) ph = 14;

  return ph;
}

// ---- pH Classification ----
String getPHLevel(float ph) {
  if (ph < 6.5)
    return "ACIDIC";
  else if (ph <= 7.5)
    return "NEUTRAL";
  else
    return "ALKALINE";
}

// ===================== SETUP =====================
void setup() {
  Serial.begin(115200);
  delay(3000);

  Serial.println("========================================");
  Serial.println("SYSTEM INITIALIZATION");
  Serial.println("========================================");

  // ---- I2C for BH1750 ----
  Wire.begin(21, 22);

  // ---- NeoPixel ----
  strip.begin();
  strip.setBrightness(50);
  strip.clear();
  strip.show();

  // ---- BH1750 ----
  if (lightMeter.begin(BH1750::CONTINUOUS_HIGH_RES_MODE)) {
    Serial.println("BH1750 detected!");
  } else {
    Serial.println("BH1750 NOT detected!");
    while (1);
  }

  // ---- FLOAT SENSOR ----
  pinMode(FLOAT_PIN, INPUT_PULLUP);
  pinMode(RELAY_FLOAT, OUTPUT);
  digitalWrite(RELAY_FLOAT, HIGH);

  // ---- MQ135 ----
  pinMode(RELAY_MQ135, OUTPUT);
  digitalWrite(RELAY_MQ135, HIGH);

  // ---- TDS ----
  pinMode(RELAY_TDS, OUTPUT);
  digitalWrite(RELAY_TDS, HIGH);

  // ---- pH ----
  pinMode(RELAY_PH, OUTPUT);
  digitalWrite(RELAY_PH, LOW);

  Serial.println("✅ System Initialized!");
  Serial.println("========================================");
}

// ===================== LOOP =====================
void loop() {

  // ================= FLOAT SENSOR =================
  int floatState = digitalRead(FLOAT_PIN);

  if (floatState == LOW) {
    Serial.println("Water Level LOW → Pump ON");
    digitalWrite(RELAY_FLOAT, LOW);
  } else {
    Serial.println("Water Level OK → Pump OFF");
    digitalWrite(RELAY_FLOAT, HIGH);
  }

  // ================= MQ135 SENSOR =================
  int mqValue = analogRead(MQ135_PIN);

  Serial.print("MQ135 Value: ");
  Serial.println(mqValue);

  if (mqValue > MQ_THRESHOLD) {
    Serial.println("GAS DETECTED → Relay ON");
    digitalWrite(RELAY_MQ135, LOW);
  } else {
    Serial.println("Air Normal → Relay OFF");
    digitalWrite(RELAY_MQ135, HIGH);
  }

  // ================= TDS SENSOR =================
  int tdsValue = analogRead(TDS_PIN);

  Serial.print("TDS Value: ");
  Serial.println(tdsValue);

  if (tdsValue > TDS_THRESHOLD) {
    Serial.println("Dirty Water → Pump ON");
    digitalWrite(RELAY_TDS, LOW);
  } else {
    Serial.println("Water Clean → Pump OFF");
    digitalWrite(RELAY_TDS, HIGH);
  }

  // ================= pH SENSOR =================
  float adc = readMedian(PH_PIN, 25);
  float voltage = adc * 3.3 / 4095.0;
  float ph = readPH();
  String status = getPHLevel(ph);

  if (status == "NEUTRAL") {
    digitalWrite(RELAY_PH, LOW);
  } else {
    digitalWrite(RELAY_PH, HIGH);
  }

  Serial.print("pH ADC: ");
  Serial.print(adc);
  Serial.print(" | Voltage: ");
  Serial.print(voltage, 3);
  Serial.print(" V");
  Serial.print(" | pH: ");
  Serial.print(ph, 2);
  Serial.print(" | Status: ");
  Serial.print(status);
  Serial.print(" | Relay: ");
  Serial.println(status == "NEUTRAL" ? "OFF" : "ON");

  // ================= BH1750 + NEOPIXEL =================
float lux = lightMeter.readLightLevel();

Serial.print("Light: ");
Serial.print(lux);
Serial.print(" lx  -> ");

int newColor;

if (lux <= 10) {
    newColor = 0;   // WHITE
    Serial.println("WHITE");
} else if (lux <= 50) {
    newColor = 1;   // RED
    Serial.println("RED");
} else if (lux <= 80) {
    newColor = 2;   // GREEN
    Serial.println("GREEN");
} else if (lux <= 100) {
    newColor = 3;   // BLUE
    Serial.println("BLUE");
} else {
    newColor = 4;   // OFF
    Serial.println("OFF");
}

if (newColor != currentColor) {
    currentColor = newColor;

    switch (currentColor) {
        case 0: setAll(255, 255, 255); break; // White
        case 1: setAll(255, 0, 0);     break; // Red
        case 2: setAll(0, 255, 0);     break; // Green
        case 3: setAll(0, 0, 255);     break; // Blue
        case 4: setAll(0, 0, 0);       break; // Off
    }
}

  Serial.println("------------------------");

  delay(1000);
}