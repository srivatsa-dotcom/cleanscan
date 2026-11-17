// CleanScan depth module clip: holds a MaixSense A010 next to the phone's rear camera and
// an ESP32-S3 board on top. Print in PETG or PLA, 0.2 mm layers, 30 % infill, no supports
// (print with the back plate flat on the bed).
//
// Measure YOUR phone and boards and change the parameters below; the defaults fit an
// iPhone 11 in a thin case, a MaixSense A010 and an ESP32-S3-DevKitC-1.
//
// SPDX-License-Identifier: CERN-OHL-P-2.0
// Copyright 2026 CleanScan contributors

/* [Phone] */
phone_thickness   = 9.8;   // mm, including the case
camera_from_top   = 22;    // mm, phone's top edge to the centre of the main rear camera
camera_from_side  = 22;    // mm, phone's side edge to the camera centre
grip_depth        = 14;    // mm the jaws reach over the front and back of the phone
clip_width        = 46;    // mm along the phone's top edge

/* [A010 depth camera] */
a010_w            = 40;    // mm (measure your module)
a010_h            = 38;    // mm
a010_lens_d       = 14;    // mm, opening for the lens (with margin)
a010_lens_dx      = 0;     // mm, lens offset from the module centre
a010_hole_d       = 2.6;   // M2.5 mounting holes
a010_hole_dx      = 34;    // hole spacing, x
a010_hole_dy      = 32;    // hole spacing, y

/* [ESP32-S3 board] */
esp_l             = 70;    // mm (DevKitC-1: 69 x 25.5; XIAO ESP32S3: 21 x 17.5)
esp_w             = 26.5;
esp_rail          = 2;     // mm, rails the board slides into

/* [Print] */
wall              = 2.4;
gap               = 0.3;   // printing tolerance
$fn               = 48;

back_h = camera_from_top + a010_h / 2 + 6;  // the back plate reaches down past the camera

module rounded_box(size, r = 1.5) {
    hull() for (x = [r, size[0] - r], y = [r, size[1] - r])
        translate([x, y, 0]) cylinder(r = r, h = size[2]);
}

// C-shaped jaw over the phone's top edge (phone lies in the x-y plane, top edge along x)
module jaw() {
    t = phone_thickness + 2 * gap;
    difference() {
        rounded_box([clip_width, back_h, t + 2 * wall]);
        translate([-1, wall, wall]) cube([clip_width + 2, back_h + 1, t]);  // phone slot below the top wall
        // front lip only reaches grip_depth over the screen
        translate([-1, grip_depth, wall + t - 0.01]) cube([clip_width + 2, back_h, wall + 1]);
    }
}

// A010 holder on the back plate, lens next to the phone camera (same direction)
module a010_mount() {
    cx = camera_from_side + a010_w / 2 + 4;  // module beside the camera, 4 mm gap
    cy = camera_from_top;
    translate([cx - a010_w / 2 - wall, cy - a010_h / 2 - wall, -wall])
        difference() {
            rounded_box([a010_w + 2 * wall, a010_h + 2 * wall, wall]);
            translate([wall + a010_w / 2 + a010_lens_dx, wall + a010_h / 2, -1])
                cylinder(d = a010_lens_d, h = wall + 2);
            for (sx = [-1, 1], sy = [-1, 1])
                translate([wall + a010_w / 2 + sx * a010_hole_dx / 2, wall + a010_h / 2 + sy * a010_hole_dy / 2, -1])
                    cylinder(d = a010_hole_d, h = wall + 2);
        }
}

// rails on top of the jaw for the ESP32 board (USB connector facing the side)
module esp_rails() {
    t = phone_thickness + 2 * gap + 2 * wall;
    translate([(clip_width - esp_l) / 2, -esp_w - wall, 0])
        difference() {
            cube([esp_l, esp_w + 2 * wall, t]);
            translate([-1, wall, esp_rail]) cube([esp_l + 2, esp_w + gap, t]);
            translate([-1, wall + esp_rail, -1]) cube([esp_l + 2, esp_w - 2 * esp_rail, esp_rail + 2]);
        }
}

jaw();
a010_mount();
esp_rails();
