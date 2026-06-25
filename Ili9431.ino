#include <WiFi.h>
#include <Adafruit_GFX.h>
#include <Adafruit_ILI9341.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// Kết nối màn hình TFT
#define TFT_CS     15
#define TFT_RST    2
#define TFT_DC     4

Adafruit_ILI9341 tft = Adafruit_ILI9341(TFT_CS, TFT_DC, TFT_RST);

// WiFi thông tin
const char* WIFI_SSID = "Cuu toi mon tn2";
const char* WIFI_PASS = "cuutoivoi";

// HTTP server
WebServer server(80);

// Biến lưu trữ thông tin nhận dạng
String objectLabel = "";
String objectConfidence = "";

// Xử lý dữ liệu nhận từ Python
void handleUpdate() {
  if (server.hasArg("plain")) {
    String body = server.arg("plain");
    Serial.println("Received data: " + body);

    // Parse JSON
    StaticJsonDocument<200> doc;
    DeserializationError error = deserializeJson(doc, body);
    if (error) {
      Serial.println("JSON Parse Error");
      server.send(400, "text/plain", "Invalid JSON");
      return;
    }

    objectLabel = doc["label"].as<String>();
    objectConfidence = doc["confidence"].as<String>();

    Serial.println("Label: " + objectLabel);
    Serial.println("Confidence: " + objectConfidence);

    updateDisplay(objectLabel, objectConfidence);
    server.send(200, "text/plain", "Data received");
  } else {
    Serial.println("No data received");
    server.send(400, "text/plain", "No data received");
  }
}

// Hiển thị thông tin lên màn hình TFT
void updateDisplay(String label, String confidence) {
  Serial.println("Updating display...");
  tft.fillScreen(ILI9341_BLACK);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 20);
  tft.print("Object: ");
  tft.print(label);

  tft.setCursor(10, 50);
  tft.print("Confidence: ");
  tft.print(confidence);
  tft.print("%");
}

void setup() {
  Serial.begin(115200);

  // Khởi tạo màn hình TFT
  tft.begin();
  tft.setRotation(1);
  tft.fillScreen(ILI9341_BLACK);
  tft.setTextColor(ILI9341_WHITE);
  tft.setTextSize(2);
  tft.setCursor(10, 10);
  tft.print("Waiting for data...");

  // Kết nối WiFi
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.println(WiFi.localIP());

  // HTTP server
  server.on("/update", HTTP_POST, handleUpdate);
  server.begin();
}

void loop() {
  server.handleClient();
}
