import ARKit
import Foundation

/// Writes the high-rate streams of a scan (docs/capture_format.md):
///   poses.bin   every ARKit pose (~60 Hz): f64 t + 16 x f32 camera_to_world, row-major
///   lidar.bin   rig readings on the phone clock: f64 t, f32 angle, f32 range, u8 intensity
///   rig.json    sensor description + clock sync quality (only when a rig was used)
///   planes.json ARKit plane anchors at the end of the scan (help the rig calibration)
final class StreamRecorder {
    private let directory: URL
    private let queue = DispatchQueue(label: "cleanscan.streams")
    private let poses: FileHandle
    private var lidar: FileHandle?
    private var lidarCount = 0

    init(directory: URL) throws {
        self.directory = directory
        poses = try Self.handle(directory.appendingPathComponent("poses.bin"))
    }

    private static func handle(_ url: URL) throws -> FileHandle {
        FileManager.default.createFile(atPath: url.path, contents: nil)
        return try FileHandle(forWritingTo: url)
    }

    func addPose(t: TimeInterval, cameraToWorld m: simd_float4x4) {
        var d = Data(capacity: 72)
        var tt = t
        d.append(Data(bytes: &tt, count: 8))
        for r in 0..<4 {
            for c in 0..<4 {
                var v = m[c][r]  // simd is column-major; the file is row-major
                d.append(Data(bytes: &v, count: 4))
            }
        }
        queue.async { try? self.poses.write(contentsOf: d) }
    }

    func addLidar(_ readings: [RigReading]) {
        var d = Data(capacity: readings.count * 17)
        for r in readings {
            var t = r.t, a = r.angle, rg = r.range, i = r.intensity
            d.append(Data(bytes: &t, count: 8))
            d.append(Data(bytes: &a, count: 4))
            d.append(Data(bytes: &rg, count: 4))
            d.append(Data(bytes: &i, count: 1))
        }
        queue.async {
            if self.lidar == nil {
                self.lidar = try? Self.handle(self.directory.appendingPathComponent("lidar.bin"))
            }
            try? self.lidar?.write(contentsOf: d)
            self.lidarCount += readings.count
        }
    }

    func finish(rig: [String: Any]?, planes: [ARPlaneAnchor], completion: @escaping () -> Void) {
        let planeJSON: [[String: Any]] = planes.map { p in
            var m: [Float] = []
            for r in 0..<4 { for c in 0..<4 { m.append(p.transform[c][r]) } }
            return [
                "transform": m,
                "center": [p.center.x, p.center.y, p.center.z],
                // the extent may be rotated about the plane normal: store a square that covers it
                "extent": [max(p.planeExtent.width, p.planeExtent.height), 0, max(p.planeExtent.width, p.planeExtent.height)],
                "alignment": p.alignment == .horizontal ? "horizontal" : "vertical",
            ]
        }
        queue.async {
            try? self.poses.close()
            try? self.lidar?.close()
            if var rig, self.lidarCount > 0 {
                rig["readings"] = self.lidarCount
                if let d = try? JSONSerialization.data(withJSONObject: rig, options: [.prettyPrinted, .sortedKeys]) {
                    try? d.write(to: self.directory.appendingPathComponent("rig.json"))
                }
            }
            if !planeJSON.isEmpty, let d = try? JSONSerialization.data(withJSONObject: planeJSON) {
                try? d.write(to: self.directory.appendingPathComponent("planes.json"))
            }
            completion()
        }
    }
}
