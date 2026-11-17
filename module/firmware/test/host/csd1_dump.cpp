// Writes the CSD1 packets of a known frame, one per file, for rig/tests (Python decoder check).
// SPDX-License-Identifier: MIT
#include <cstdio>
#include <string>

#include "cleanscan_proto.h"

int main(int argc, char** argv) {
  const std::string dir = argc > 1 ? argv[1] : ".";
  cleanscan::DepthFrame f;
  f.frame_id = 4242;
  f.t_capture = 1000.125;
  f.width = 100;
  f.height = 100;
  f.mm.resize(10000);
  for (size_t i = 0; i < f.mm.size(); ++i) f.mm[i] = uint16_t((i * 37) % 2600);
  int k = 0;
  for (auto& p : cleanscan::encode_depth_frame(f)) {
    std::FILE* o = std::fopen((dir + "/pkt" + std::to_string(k++) + ".bin").c_str(), "wb");
    std::fwrite(p.data(), 1, p.size(), o);
    std::fclose(o);
  }
  std::printf("%d\n", k);
  return 0;
}
