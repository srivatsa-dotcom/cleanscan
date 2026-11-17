// SPDX-License-Identifier: MIT
#include "cleanscan_proto.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace cleanscan {

namespace {
void put_u16(std::vector<uint8_t>& b, uint16_t v) { b.push_back(v & 0xFF); b.push_back(v >> 8); }
void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; ++i) b.push_back((v >> (8 * i)) & 0xFF); }
void put_f64(std::vector<uint8_t>& b, double v) {
  uint64_t u;
  std::memcpy(&u, &v, 8);  // IEEE-754; ESP32 and x86 are both little-endian
  for (int i = 0; i < 8; ++i) b.push_back((u >> (8 * i)) & 0xFF);
}
}  // namespace

std::vector<std::vector<uint8_t>> encode_depth_frame(const DepthFrame& f) {
  const size_t bytes = size_t(f.width) * f.height * 2;
  const size_t n = bytes == 0 ? 1 : (bytes + kMaxDepthPayload - 1) / kMaxDepthPayload;
  std::vector<std::vector<uint8_t>> out;
  out.reserve(n);
  for (size_t k = 0; k < n; ++k) {
    std::vector<uint8_t> p;
    p.reserve(kDepthHeaderSize + kMaxDepthPayload);
    p.insert(p.end(), {'C', 'S', 'D', '1'});
    p.push_back(3);  // kind: depth frame
    p.push_back(1);  // format: u16 millimetres
    put_u16(p, f.width);
    put_u16(p, f.height);
    put_u32(p, f.frame_id);
    put_u16(p, uint16_t(k));
    put_u16(p, uint16_t(n));
    put_f64(p, f.t_capture);
    const size_t b0 = k * kMaxDepthPayload, b1 = std::min(bytes, b0 + kMaxDepthPayload);
    for (size_t b = b0; b < b1; b += 2) put_u16(p, f.mm[b / 2]);
    out.push_back(std::move(p));
  }
  return out;
}

uint16_t a010_code_to_mm(uint8_t code, uint8_t unit) {
  if (code == 0) return 0;
  if (unit == 0) {
    const double v = code / 5.1;
    return uint16_t(std::lround(v * v));
  }
  return uint16_t(code) * unit;
}

std::vector<uint8_t> a010_encode_frame(const std::vector<uint8_t>& codes, uint8_t rows, uint8_t cols,
                                       uint16_t frame_id) {
  std::vector<uint8_t> f = {0x00, 0xFF};
  const uint16_t len = uint16_t(16 + codes.size());
  put_u16(f, len);
  const uint8_t head[16] = {0xFF, 0x00, 40, 40, 0, 0, 0, 0, 0, 0x00, rows, cols,
                            uint8_t(frame_id & 0xFF), uint8_t(frame_id >> 8), 1, 0xFF};
  f.insert(f.end(), head, head + 16);
  f.insert(f.end(), codes.begin(), codes.end());
  uint32_t sum = 0;
  for (uint8_t b : f) sum += b;
  f.push_back(uint8_t(sum & 0xFF));
  f.push_back(0xDD);
  return f;
}

bool A010Parser::feed(const uint8_t* data, size_t n, DepthFrame& out) {
  buf_.insert(buf_.end(), data, data + n);
  bool got = false;
  DepthFrame tmp;
  while (parse_one(tmp)) {  // keep the newest complete frame
    out = std::move(tmp);
    got = true;
  }
  if (buf_.size() > 64 * 1024) buf_.erase(buf_.begin(), buf_.end() - 4);  // never grow without bound
  return got;
}

bool A010Parser::parse_one(DepthFrame& out) {
  for (;;) {
    size_t i = 0;
    while (i + 1 < buf_.size() && !(buf_[i] == 0x00 && buf_[i + 1] == 0xFF)) ++i;
    if (i) buf_.erase(buf_.begin(), buf_.begin() + i);
    if (buf_.size() < 4) return false;
    const size_t len = buf_[2] | (buf_[3] << 8);
    if (len < 16 || len > 16 + 320 * 240) {  // not a plausible frame: resync
      ++bad_;
      buf_.erase(buf_.begin());
      continue;
    }
    const size_t total = 4 + len + 2;
    if (buf_.size() < total) return false;
    uint32_t sum = 0;
    for (size_t k = 0; k < 4 + len; ++k) sum += buf_[k];
    const uint8_t rows = buf_[4 + 10], cols = buf_[4 + 11];
    const bool ok = (sum & 0xFF) == buf_[4 + len] && buf_[4 + len + 1] == 0xDD &&
                    size_t(rows) * cols == len - 16 && rows && cols;
    if (!ok) {
      ++bad_;
      buf_.erase(buf_.begin());
      continue;
    }
    out.width = cols;
    out.height = rows;
    out.frame_id = buf_[4 + 12] | (buf_[4 + 13] << 8);
    out.mm.resize(size_t(rows) * cols);
    for (size_t k = 0; k < out.mm.size(); ++k) out.mm[k] = a010_code_to_mm(buf_[4 + 16 + k], unit_);
    buf_.erase(buf_.begin(), buf_.begin() + total);
    ++good_;
    return true;
  }
}

std::vector<std::string> a010_setup_commands(int baud_index, int fps, uint8_t unit) {
  // AT+ISP=0 stop the sensor pipeline while configuring; AT+DISP=4: output on UART only;
  // AT+BINN=1: full 100x100; AT+UNIT: depth code unit; AT+FPS; AT+BAUD (index, see README).
  char b[32];
  std::vector<std::string> c = {"AT+ISP=0\r", "AT+DISP=4\r", "AT+BINN=1\r"};
  std::snprintf(b, sizeof b, "AT+UNIT=%u\r", unsigned(unit));
  c.push_back(b);
  std::snprintf(b, sizeof b, "AT+FPS=%d\r", fps);
  c.push_back(b);
  std::snprintf(b, sizeof b, "AT+BAUD=%d\r", baud_index);
  c.push_back(b);
  c.push_back("AT+ISP=1\r");
  return c;
}

namespace {
// Minimal extraction of "key": value from a flat JSON object (the app sends tiny messages).
const char* find_key(const std::string& s, const char* key) {
  std::string pat = std::string("\"") + key + "\"";
  size_t p = s.find(pat);
  if (p == std::string::npos) return nullptr;
  p = s.find(':', p + pat.size());
  if (p == std::string::npos) return nullptr;
  ++p;
  while (p < s.size() && (s[p] == ' ' || s[p] == '\t')) ++p;
  return s.c_str() + p;
}
}  // namespace

Control parse_control(const char* data, size_t n) {
  Control c;
  if (n == 0 || data[0] != '{') return c;
  std::string s(data, n);
  if (const char* v = find_key(s, "op")) {
    if (*v == '"') {
      const char* e = std::strchr(v + 1, '"');
      if (e) c.op.assign(v + 1, e);
    }
  }
  if (const char* v = find_key(s, "id")) c.id = std::strtol(v, nullptr, 10);
  if (const char* v = find_key(s, "t1")) {
    char* end = nullptr;
    c.t1 = std::strtod(v, &end);
    c.has_t1 = end != v;
  }
  return c;
}

std::string info_json(const char* sensor, int width, int height, double fov_h, double fov_v,
                      const char* firmware) {
  char b[256];
  std::snprintf(b, sizeof b,
                "{\"op\":\"info\",\"sensor\":\"%s\",\"kind\":\"depth\",\"version\":1,\"width\":%d,"
                "\"height\":%d,\"fov_deg\":[%.1f,%.1f],\"firmware\":\"%s\"}",
                sensor, width, height, fov_h, fov_v, firmware);
  return b;
}

std::string pong_json(long id, double t1, double t2) {
  char b[160];
  std::snprintf(b, sizeof b, "{\"op\":\"pong\",\"id\":%ld,\"t1\":%.9f,\"t2\":%.9f}", id, t1, t2);
  return b;
}

}  // namespace cleanscan
