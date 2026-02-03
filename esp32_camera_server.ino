#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <WebServer.h>

// ==== camera pins from wiki Timed camera example ====[page:1]
#define PWDN_GPIO_NUM     -1
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      5
#define Y9_GPIO_NUM        4
#define Y8_GPIO_NUM        6
#define Y7_GPIO_NUM        7
#define Y6_GPIO_NUM       14
#define Y5_GPIO_NUM       17
#define Y4_GPIO_NUM       21
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM       16
#define VSYNC_GPIO_NUM     1
#define HREF_GPIO_NUM      2
#define PCLK_GPIO_NUM     15
#define SIOD_GPIO_NUM      8
#define SIOC_GPIO_NUM      9

// ==== config ====
const char* WIFI_SSID     = "zoowifi-EXT5G";
const char* WIFI_PASSWORD = "seadogbow80";
const char* POST_URL      = "https://aiagent.maxsolo.co.uk/"; // change this

const unsigned long INTERVAL_MS = 10000; // 10 seconds

unsigned long lastShot = 0;

// ==== Web Server ====
WebServer server(8080);  // ESP32 camera server on port 8080

bool initCamera() {
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer   = LEDC_TIMER_0;
  config.pin_d0       = Y2_GPIO_NUM;
  config.pin_d1       = Y3_GPIO_NUM;
  config.pin_d2       = Y4_GPIO_NUM;
  config.pin_d3       = Y5_GPIO_NUM;
  config.pin_d4       = Y6_GPIO_NUM;
  config.pin_d5       = Y7_GPIO_NUM;
  config.pin_d6       = Y8_GPIO_NUM;
  config.pin_d7       = Y9_GPIO_NUM;
  config.pin_xclk     = XCLK_GPIO_NUM;
  config.pin_pclk     = PCLK_GPIO_NUM;
  config.pin_vsync    = VSYNC_GPIO_NUM;
  config.pin_href     = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn     = PWDN_GPIO_NUM;
  config.pin_reset    = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;
  config.frame_size   = FRAMESIZE_SVGA; // smaller than UXGA for faster upload
  config.fb_location  = CAMERA_FB_IN_PSRAM;
  config.jpeg_quality = 12;             // 0-63, lower = better quality
  config.fb_count     = 2;
  config.grab_mode    = CAMERA_GRAB_LATEST;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed 0x%x\n", err);
    return false;
  }

  sensor_t * s = esp_camera_sensor_get();
  if (s->id.PID == OV3660_PID) {        // OV3660 per wiki[page:1]
    s->set_vflip(s, 1);
    s->set_brightness(s, 1);
    s->set_saturation(s, -2);
  }
  return true;
}

void connectWiFi() {
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to WiFi %s\n", WIFI_SSID);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("\nWiFi connected");
  Serial.print("Camera server IP: ");
  Serial.println(WiFi.localIP());
}

bool sendFrame(camera_fb_t* fb) {
  if (WiFi.status() != WL_CONNECTED) {
    connectWiFi();
  }

  HTTPClient http;
  http.begin(POST_URL);
  http.addHeader("Content-Type", "image/jpeg");

  int httpCode = http.POST(fb->buf, fb->len);
  Serial.printf("HTTP POST code: %d\n", httpCode);

  http.end();
  return (httpCode == 200 || httpCode == 201);
}

// ==== Web Server Handlers ====
void handleCapture() {
  camera_fb_t *fb = esp_camera_fb_get();
  if (!fb) {
    server.send(500, "text/plain", "Failed to get frame");
    Serial.println("Failed to get frame for /capture");
    return;
  }

  server.sendHeader("Content-Type", "image/jpeg");
  server.sendHeader("Content-Length", String(fb->len));
  server.send(200, "image/jpeg", (const char *)fb->buf, fb->len);
  
  Serial.printf("Image sent: %d bytes\n", fb->len);
  esp_camera_fb_return(fb);
}

void handleStatus() {
  String status = "{\"status\": \"ok\", \"ip\": \"";
  status += WiFi.localIP().toString();
  status += "\", \"ssid\": \"";
  status += WiFi.SSID();
  status += "\"}";
  
  server.sendHeader("Content-Type", "application/json");
  server.send(200, "application/json", status);
}

void handleRoot() {
  String html = R"(
    <html>
    <head>
      <title>ESP32 Camera Server</title>
      <style>
        body { font-family: Arial; text-align: center; margin-top: 50px; }
        h1 { color: #333; }
        .status { background: #e8f5e9; padding: 20px; border-radius: 5px; margin: 20px; }
        .url { background: #fff9c4; padding: 10px; margin: 10px; font-family: monospace; word-break: break-all; }
      </style>
    </head>
    <body>
      <h1>✅ ESP32 Camera Server Running</h1>
      <div class="status">
        <p><strong>IP Address:</strong> )";
  html += WiFi.localIP().toString();
  html += R"(</p>
        <p><strong>WiFi SSID:</strong> )";
  html += WiFi.SSID();
  html += R"(</p>
      </div>
      <h2>Endpoints:</h2>
      <div class="url">GET /capture - Fetch latest JPEG image</div>
      <div class="url">GET /status - Get server status (JSON)</div>
      <p style="margin-top: 40px; color: #999; font-size: 12px;">
        Use this IP in your HTML form: <strong>http://)" + WiFi.localIP().toString() + R"(:8080</strong>
      </p>
    </body>
    </html>
  )";
  
  server.sendHeader("Content-Type", "text/html");
  server.send(200, "text/html", html);
}

void setup() {
  Serial.begin(115200);
  delay(3000);

  connectWiFi();
  if (!initCamera()) {
    Serial.println("Camera init failed");
    while (true) { delay(1000); }
  }

  // Setup web server routes
  server.on("/", handleRoot);
  server.on("/capture", HTTP_GET, handleCapture);
  server.on("/status", HTTP_GET, handleStatus);
  
  server.begin();
  Serial.println("Web server started on port 8080");

  lastShot = millis();
}

void loop() {
  server.handleClient();  // Handle incoming web requests
  
  unsigned long now = millis();
  if (now - lastShot >= INTERVAL_MS) {
    lastShot = now;

    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
      Serial.println("Failed to get frame");
      return;
    }

    bool ok = sendFrame(fb);
    Serial.println(ok ? "Upload OK" : "Upload FAILED");

    esp_camera_fb_return(fb);
  }

  delay(10);
}
