// Host tests of the portable protocol code: `pio test -e native`, or without PlatformIO:
//   g++ -std=c++17 -Ilib/cleanscan_proto/src test/test_native/test_main.cpp lib/cleanscan_proto/src/cleanscan_proto.cpp
// SPDX-License-Identifier: MIT
#include <cmath>
#include <cstdio>
#include <cstring>

#include "cleanscan_proto.h"

#if __has_include(<unity.h>)
#include <unity.h>
#define CHECK(x) TEST_ASSERT_TRUE(x)
#else
static int g_failed = 0;
#define CHECK(x) do { if (!(x)) { std::printf("FAIL %s:%d %s\n", __FILE__, __LINE__, #x); ++g_failed; } } while (0)
#define RUN_TEST(f) do { f(); std::printf("ran %s\n", #f); } while (0)
#endif

using namespace cleanscan;

void setUp() {}
void tearDown() {}

static void test_a010_units() {
  CHECK(a010_code_to_mm(0, 0) == 0);
  CHECK(a010_code_to_mm(255, 0) == 2500);  // (255 / 5.1)^2: the sensor's 2.5 m range
  CHECK(a010_code_to_mm(51, 0) == 100);
  CHECK(a010_code_to_mm(100, 5) == 500);
}

static void test_a010_parser_resyncs_and_checks() {
  std::vector<uint8_t> codes(100 * 100);
  for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(1 + i % 250);
  auto good = a010_encode_frame(codes, 100, 100, 7);
  auto bad = good;
  bad[30] ^= 0x5A;  // corrupt payload -> checksum mismatch
  std::vector<uint8_t> stream = {0x12, 0x00, 0xFF, 0x34};  // garbage incl. a false sync
  stream.insert(stream.end(), bad.begin(), bad.end());
  stream.insert(stream.end(), good.begin(), good.end());
  A010Parser p(0);
  DepthFrame f;
  int frames = 0;
  for (size_t i = 0; i < stream.size(); i += 333) {  // arbitrary UART chunking
    if (p.feed(stream.data() + i, std::min<size_t>(333, stream.size() - i), f)) ++frames;
  }
  CHECK(frames == 1);
  CHECK(p.bad_frames() >= 1);
  CHECK(f.width == 100 && f.height == 100 && f.frame_id == 7);
  CHECK(f.mm[0] == a010_code_to_mm(1, 0) && f.mm[9999] == a010_code_to_mm(uint8_t(1 + 9999 % 250), 0));
}

static void test_csd1_packets() {
  DepthFrame f;
  f.frame_id = 42;
  f.t_capture = 123.456;
  f.width = 100;
  f.height = 100;
  f.mm.resize(10000);
  for (size_t i = 0; i < f.mm.size(); ++i) f.mm[i] = uint16_t(i * 3);
  auto pk = encode_depth_frame(f);
  CHECK(pk.size() == 15);
  size_t total = 0;
  for (auto& p : pk) {
    CHECK(p.size() <= kMaxDatagram);
    CHECK(std::memcmp(p.data(), "CSD1", 4) == 0);
    total += p.size() - kDepthHeaderSize;
  }
  CHECK(total == 20000);
  double t;
  std::memcpy(&t, pk[3].data() + 18, 8);
  CHECK(t == 123.456);
  CHECK(pk[14][14] == 14 && pk[14][16] == 15);  // chunk index, chunk count
}

static void test_control_messages() {
  const char* ping = "{\"op\": \"ping\", \"id\": 7, \"t1\": 1234.5678}";
  Control c = parse_control(ping, std::strlen(ping));
  CHECK(c.op == "ping" && c.id == 7 && c.has_t1 && std::fabs(c.t1 - 1234.5678) < 1e-9);
  CHECK(parse_control("CSL1....", 8).op.empty());
  std::string pong = pong_json(7, 1234.5678, 99.25);
  CHECK(pong.find("\"op\":\"pong\"") != std::string::npos && pong.find("\"t2\":99.25") != std::string::npos);
  std::string info = info_json("a010", 100, 100, 70, 60, "1.0.0");
  CHECK(info.find("\"kind\":\"depth\"") != std::string::npos);
  auto cmds = a010_setup_commands(7, 15, 0);
  CHECK(cmds.front() == "AT+ISP=0\r" && cmds.back() == "AT+ISP=1\r");
}

int main() {
#if __has_include(<unity.h>)
  UNITY_BEGIN();
#endif
  RUN_TEST(test_a010_units);
  RUN_TEST(test_a010_parser_resyncs_and_checks);
  RUN_TEST(test_csd1_packets);
  RUN_TEST(test_control_messages);
#if __has_include(<unity.h>)
  return UNITY_END();
#else
  std::printf(g_failed ? "%d FAILED\n" : "all passed\n", g_failed);
  return g_failed ? 1 : 0;
#endif
}
