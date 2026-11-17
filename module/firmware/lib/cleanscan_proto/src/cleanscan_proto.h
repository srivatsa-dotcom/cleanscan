// CleanScan depth module: portable protocol code (no ESP-IDF), unit-tested on the host.
// SPDX-License-Identifier: MIT
//
// * MaixSense A010 UART frame parser (Sipeed "frame_head_t" layout) and depth decoding
// * CSD1 depth-frame packets, identical to rig/cleanscan_rig/protocol.py
// * the JSON control messages of the rig protocol (hello / info / ping / pong)
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace cleanscan {

constexpr uint16_t kPort = 5600;
constexpr size_t kMaxDatagram = 1400;
constexpr size_t kDepthHeaderSize = 4 + 1 + 1 + 2 + 2 + 4 + 2 + 2 + 8;  // = 26, "<4sBBHHIHHd"
constexpr size_t kMaxDepthPayload = (kMaxDatagram - kDepthHeaderSize) & ~size_t(1);

struct DepthFrame {
  uint32_t frame_id = 0;
  double t_capture = 0;  // module clock, seconds
  uint16_t width = 0, height = 0;
  std::vector<uint16_t> mm;  // row-major, 0 = no measurement
};

// Split one depth frame into CSD1 UDP datagrams (little-endian, as the Python decoder expects).
std::vector<std::vector<uint8_t>> encode_depth_frame(const DepthFrame& f);

// --------------------------------------------------------------------------- A010

// A010 depth code -> millimetres. Unit 0 ("auto"): mm = (code / 5.1)^2, reaching 2500 mm at 255.
// Unit 1..10: mm = code * unit.   (Sipeed documentation; verify against your firmware version.)
uint16_t a010_code_to_mm(uint8_t code, uint8_t unit);

// Incremental parser of the A010 UART stream:
//   0x00 0xFF | u16 len | 16-byte frame header | rows*cols depth codes | u8 checksum | 0xDD
// len counts the header + depth bytes; the checksum is the low byte of the sum of every byte
// before it. Garbage and corrupt frames are skipped; feed() takes arbitrary chunks.
class A010Parser {
 public:
  explicit A010Parser(uint8_t unit = 0) : unit_(unit) {}
  // Returns true when `out` holds a new frame (t_capture is set by the caller).
  bool feed(const uint8_t* data, size_t n, DepthFrame& out);
  uint32_t bad_frames() const { return bad_; }
  uint32_t frames() const { return good_; }

 private:
  bool parse_one(DepthFrame& out);
  std::vector<uint8_t> buf_;
  uint8_t unit_;
  uint32_t bad_ = 0, good_ = 0;
};

// Build one A010 frame (for tests and the host simulator).
std::vector<uint8_t> a010_encode_frame(const std::vector<uint8_t>& codes, uint8_t rows, uint8_t cols,
                                       uint16_t frame_id);

// AT commands that put the A010 in UART streaming mode (each ends with "\r").
std::vector<std::string> a010_setup_commands(int baud_index, int fps, uint8_t unit);

// --------------------------------------------------------------------------- control messages

struct Control {
  std::string op;  // "hello", "ping", or "" if not a control message
  long id = 0;
  double t1 = 0;
  bool has_t1 = false;
};

Control parse_control(const char* data, size_t n);
std::string info_json(const char* sensor, int width, int height, double fov_h, double fov_v,
                      const char* firmware);
std::string pong_json(long id, double t1, double t2);

}  // namespace cleanscan
