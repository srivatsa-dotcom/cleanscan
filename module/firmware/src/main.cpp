// CleanScan depth module firmware: ESP32-S3 + Sipeed MaixSense A010 ToF camera.
// Streams timestamped depth frames to the CleanScan app over the module's own Wi-Fi.
//
//   Wi-Fi access point "CleanScanDepth" (password: menuconfig / build flag, changeable over
//   USB serial with `pass <new password>`), module address 192.168.4.1, UDP port 5600.
//   Protocol: rig/cleanscan_rig/protocol.py (hello/info, ping/pong clock sync, CSD1 frames).
//
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "cleanscan_proto.h"
#include "driver/gpio.h"
#include "driver/uart.h"
#include "esp_event.h"
#include "esp_log.h"
#include "esp_mac.h"
#include "esp_netif.h"
#include "esp_timer.h"
#include "esp_wifi.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "nvs.h"
#include "nvs_flash.h"

#ifndef CLEANSCAN_AP_PASSWORD
#define CLEANSCAN_AP_PASSWORD "cleanscan123"
#endif
#ifndef CLEANSCAN_A010_TX
#define CLEANSCAN_A010_TX 17  // ESP32 TX -> A010 RX
#endif
#ifndef CLEANSCAN_A010_RX
#define CLEANSCAN_A010_RX 18  // ESP32 RX <- A010 TX
#endif
#define FIRMWARE_VERSION "1.0.0"

static const char* TAG = "cleanscan";
static constexpr uart_port_t kUart = UART_NUM_1;
// A010 AT+BAUD index 7 = 2 Mbit/s: a 100x100 frame (~10 kB) takes ~50 ms, enough for 15 fps.
// (index table from Sipeed's docs: 0=9600 1=57600 2=115200 3=230400 4=460800 5=921600 6=1M 7=2M 8=3M)
static constexpr int kBaudIndex = 7;
static constexpr int kStreamBaud = 2000000;
static constexpr int kFps = 15;
static constexpr double kClientTimeout = 5.0;

static double now_s() { return esp_timer_get_time() / 1e6; }

// ------------------------------------------------------------------ subscribers

struct Client {
  sockaddr_in addr;
  double seen;
};
static std::vector<Client> g_clients;
static SemaphoreHandle_t g_lock;
static int g_sock = -1;

static void touch_client(const sockaddr_in& a) {
  xSemaphoreTake(g_lock, portMAX_DELAY);
  bool found = false;
  for (auto& c : g_clients) {
    if (c.addr.sin_addr.s_addr == a.sin_addr.s_addr && c.addr.sin_port == a.sin_port) {
      c.seen = now_s();
      found = true;
    }
  }
  if (!found) {
    g_clients.push_back({a, now_s()});
    ESP_LOGI(TAG, "app connected");
  }
  xSemaphoreGive(g_lock);
}

// ------------------------------------------------------------------ Wi-Fi

static std::string load_password() {
  nvs_handle_t h;
  char buf[64] = CLEANSCAN_AP_PASSWORD;
  size_t len = sizeof buf;
  if (nvs_open("cleanscan", NVS_READONLY, &h) == ESP_OK) {
    if (nvs_get_str(h, "ap_pass", buf, &len) != ESP_OK) std::strcpy(buf, CLEANSCAN_AP_PASSWORD);
    nvs_close(h);
  }
  return buf;
}

static void save_password(const std::string& p) {
  nvs_handle_t h;
  if (nvs_open("cleanscan", NVS_READWRITE, &h) == ESP_OK) {
    nvs_set_str(h, "ap_pass", p.c_str());
    nvs_commit(h);
    nvs_close(h);
  }
}

static void start_access_point() {
  ESP_ERROR_CHECK(esp_netif_init());
  ESP_ERROR_CHECK(esp_event_loop_create_default());
  esp_netif_create_default_wifi_ap();
  wifi_init_config_t init = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&init));
  wifi_config_t cfg = {};
  const std::string pass = load_password();
  std::strcpy(reinterpret_cast<char*>(cfg.ap.ssid), "CleanScanDepth");
  cfg.ap.ssid_len = std::strlen("CleanScanDepth");
  std::strncpy(reinterpret_cast<char*>(cfg.ap.password), pass.c_str(), sizeof cfg.ap.password - 1);
  cfg.ap.channel = 6;
  cfg.ap.max_connection = 2;
  cfg.ap.authmode = pass.size() >= 8 ? WIFI_AUTH_WPA2_PSK : WIFI_AUTH_OPEN;
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_AP));
  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &cfg));
  ESP_ERROR_CHECK(esp_wifi_start());
  esp_wifi_set_ps(WIFI_PS_NONE);  // no power save: steady latency for the clock sync
  ESP_LOGI(TAG, "Wi-Fi \"CleanScanDepth\" up, module at 192.168.4.1:%d", cleanscan::kPort);
}

// ------------------------------------------------------------------ UDP control

static void udp_task(void*) {
  g_sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  sockaddr_in bind_addr = {};
  bind_addr.sin_family = AF_INET;
  bind_addr.sin_port = htons(cleanscan::kPort);
  bind_addr.sin_addr.s_addr = htonl(INADDR_ANY);
  bind(g_sock, reinterpret_cast<sockaddr*>(&bind_addr), sizeof bind_addr);
  char buf[512];
  for (;;) {
    sockaddr_in from = {};
    socklen_t flen = sizeof from;
    int n = recvfrom(g_sock, buf, sizeof buf, 0, reinterpret_cast<sockaddr*>(&from), &flen);
    const double t2 = now_s();  // as early as possible for the clock sync
    if (n <= 0) continue;
    cleanscan::Control c = cleanscan::parse_control(buf, size_t(n));
    if (c.op == "hello") {
      touch_client(from);
      std::string info = cleanscan::info_json("a010", 100, 100, 70.0, 60.0, FIRMWARE_VERSION);
      sendto(g_sock, info.data(), info.size(), 0, reinterpret_cast<sockaddr*>(&from), flen);
    } else if (c.op == "ping") {
      touch_client(from);
      std::string pong = cleanscan::pong_json(c.id, c.t1, t2);
      sendto(g_sock, pong.data(), pong.size(), 0, reinterpret_cast<sockaddr*>(&from), flen);
    }
  }
}

static void send_frame(const cleanscan::DepthFrame& f) {
  if (g_sock < 0) return;
  xSemaphoreTake(g_lock, portMAX_DELAY);
  const double now = now_s();
  std::vector<sockaddr_in> to;
  for (auto it = g_clients.begin(); it != g_clients.end();) {
    if (now - it->seen > kClientTimeout) {
      ESP_LOGI(TAG, "app gone");
      it = g_clients.erase(it);
    } else {
      to.push_back(it->addr);
      ++it;
    }
  }
  xSemaphoreGive(g_lock);
  if (to.empty()) return;
  for (auto& pkt : cleanscan::encode_depth_frame(f)) {
    for (auto& a : to) sendto(g_sock, pkt.data(), pkt.size(), 0, reinterpret_cast<const sockaddr*>(&a), sizeof a);
  }
}

// ------------------------------------------------------------------ A010 on the UART

static void a010_configure() {
  uart_config_t cfg = {};
  cfg.baud_rate = 115200;  // the A010's default
  cfg.data_bits = UART_DATA_8_BITS;
  cfg.parity = UART_PARITY_DISABLE;
  cfg.stop_bits = UART_STOP_BITS_1;
  cfg.flow_ctrl = UART_HW_FLOWCTRL_DISABLE;
  cfg.source_clk = UART_SCLK_DEFAULT;
  ESP_ERROR_CHECK(uart_driver_install(kUart, 32 * 1024, 0, 0, nullptr, 0));
  ESP_ERROR_CHECK(uart_param_config(kUart, &cfg));
  ESP_ERROR_CHECK(uart_set_pin(kUart, CLEANSCAN_A010_TX, CLEANSCAN_A010_RX, UART_PIN_NO_CHANGE, UART_PIN_NO_CHANGE));
  vTaskDelay(pdMS_TO_TICKS(1500));  // the A010 boots for about a second
  for (const auto& cmd : cleanscan::a010_setup_commands(kBaudIndex, kFps, 0)) {
    uart_write_bytes(kUart, cmd.data(), cmd.size());
    vTaskDelay(pdMS_TO_TICKS(150));
    if (cmd.rfind("AT+BAUD", 0) == 0) {  // the sensor switches right after answering
      vTaskDelay(pdMS_TO_TICKS(100));
      uart_set_baudrate(kUart, kStreamBaud);
    }
  }
  uart_flush_input(kUart);
  ESP_LOGI(TAG, "A010 streaming 100x100 at %d fps, %d baud", kFps, kStreamBaud);
}

static void a010_task(void*) {
  a010_configure();
  cleanscan::A010Parser parser(0);
  cleanscan::DepthFrame frame;
  std::vector<uint8_t> buf(4096);
  uint32_t sent = 0;
  double last_log = now_s();
  // a 100x100 frame (~10 kB) needs this long on the wire; the frame was captured before that
  const double transfer = (16 + 4 + 2 + 100 * 100) * 10.0 / kStreamBaud;
  for (;;) {
    int n = uart_read_bytes(kUart, buf.data(), buf.size(), pdMS_TO_TICKS(20));
    if (n <= 0) continue;
    const double t = now_s();
    if (parser.feed(buf.data(), size_t(n), frame)) {
      frame.t_capture = t - transfer;
      send_frame(frame);
      ++sent;
    }
    if (t - last_log > 10) {
      ESP_LOGI(TAG, "%.1f frames/s, %u bad frames", sent / (t - last_log), unsigned(parser.bad_frames()));
      sent = 0;
      last_log = t;
    }
  }
}

// ------------------------------------------------------------------ USB serial console

static void console_task(void*) {
  // `pass <new password>` (8+ characters) stores a new Wi-Fi password and restarts
  std::string line;
  for (;;) {
    int ch = getchar();
    if (ch == EOF) {
      vTaskDelay(pdMS_TO_TICKS(50));
      continue;
    }
    if (ch == '\n' || ch == '\r') {
      if (line.rfind("pass ", 0) == 0 && line.size() >= 5 + 8) {
        save_password(line.substr(5));
        printf("password saved, restarting\n");
        vTaskDelay(pdMS_TO_TICKS(200));
        esp_restart();
      } else if (!line.empty()) {
        printf("commands: pass <new Wi-Fi password, 8+ characters>\n");
      }
      line.clear();
    } else {
      line += char(ch);
    }
  }
}

extern "C" void app_main() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    nvs_flash_erase();
    nvs_flash_init();
  }
  g_lock = xSemaphoreCreateMutex();
  start_access_point();
  xTaskCreate(udp_task, "udp", 4096, nullptr, 6, nullptr);
  xTaskCreate(a010_task, "a010", 8192, nullptr, 5, nullptr);
  xTaskCreate(console_task, "console", 4096, nullptr, 2, nullptr);
}
