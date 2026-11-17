import CoreGraphics
import Foundation
import ImageIO
import Network
import QuartzCore
import UniformTypeIdentifiers

/// One depth image from the clip-on module (module/), already on the phone's clock.
struct ModuleDepthFrame {
    let t: Double
    let width: Int
    let height: Int
    let mm: [UInt16]
}

/// Talks to the clip-on depth module (ESP32-S3 + MaixSense A010, or a Pi + Arducam ToF): finds
/// it, keeps its clock in sync with the phone's and reassembles its CSD1 depth frames
/// (protocol: rig/cleanscan_rig/protocol.py).
///
/// Discovery: iOS apps need Apple's multicast entitlement for broadcasts, which a sideloaded
/// app cannot get, so the client says hello to the usual module addresses at once and keeps
/// the one that answers as a depth module.
final class DepthModuleClient: ObservableObject {
    @Published private(set) var connected = false
    @Published private(set) var sensor = ""
    @Published private(set) var host = ""
    @Published private(set) var framesPerSecond = 0
    @Published private(set) var rttMs: Double?
    private(set) var width = 0, height = 0
    private(set) var fovDeg: (Double, Double) = (70, 60)

    /// Called on the client's queue for every complete frame.
    var onFrame: ((ModuleDepthFrame) -> Void)?

    static let port: UInt16 = 5600
    static let defaultHosts = ["192.168.4.1", "10.42.0.1"]  // ESP32 access point, Pi hotspot
    private let queue = DispatchQueue(label: "cleanscan.depthmodule")
    private var probes: [String: NWConnection] = [:]
    private var connection: NWConnection?
    private var timer: DispatchSourceTimer?
    private var sync: [(rtt: Double, offset: Double)] = []
    private var pingId = 0
    private var lastSeen = 0.0
    private var counted = 0
    private var countStart = CACurrentMediaTime()
    private var parts: [UInt32: (n: Int, t: Double, w: Int, h: Int, chunks: [Int: Data])] = [:]

    var clockOffset: Double? { sync.min(by: { $0.rtt < $1.rtt })?.offset }

    func start(preferredHost: String) {
        stop()
        queue.async {
            var hosts = [preferredHost] + Self.defaultHosts
            hosts = hosts.filter { !$0.isEmpty }.reduce(into: []) { if !$0.contains($1) { $0.append($1) } }
            for h in hosts {
                let c = NWConnection(host: NWEndpoint.Host(h), port: NWEndpoint.Port(rawValue: Self.port)!, using: .udp)
                self.probes[h] = c
                c.start(queue: self.queue)
                self.receive(on: c, host: h)
            }
            let t = DispatchSource.makeTimerSource(queue: self.queue)
            t.schedule(deadline: .now(), repeating: 0.5)
            t.setEventHandler { [weak self] in self?.tick() }
            t.resume()
            self.timer = t
        }
    }

    func stop() {
        queue.async {
            self.timer?.cancel()
            self.timer = nil
            self.probes.values.forEach { $0.cancel() }
            self.probes.removeAll()
            self.connection = nil
            self.sync.removeAll()
            self.parts.removeAll()
        }
        DispatchQueue.main.async { self.connected = false; self.framesPerSecond = 0; self.rttMs = nil }
    }

    private func tick() {
        let targets = connection.map { [$0] } ?? Array(probes.values)
        for c in targets { send(["op": "hello"], on: c) }
        if let c = connection {
            pingId += 1
            send(["op": "ping", "id": pingId, "t1": CACurrentMediaTime()], on: c)
        }
        let now = CACurrentMediaTime()
        let rate = Int((Double(counted) / max(now - countStart, 1e-3)).rounded())
        counted = 0
        countStart = now
        let alive = now - lastSeen < 3
        let rtt = sync.min(by: { $0.rtt < $1.rtt }).map { $0.rtt * 1000 }
        DispatchQueue.main.async {
            self.framesPerSecond = rate
            self.connected = alive
            self.rttMs = rtt
        }
    }

    private func send(_ obj: [String: Any], on c: NWConnection) {
        guard let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        c.send(content: data, completion: .contentProcessed { _ in })
    }

    private func receive(on c: NWConnection, host: String) {
        c.receiveMessage { [weak self] data, _, _, error in
            guard let self, self.probes[host] === c else { return }
            if let data, !data.isEmpty { self.handle(data, from: c, host: host) }
            if error == nil { self.receive(on: c, host: host) }
        }
    }

    private func handle(_ data: Data, from c: NWConnection, host: String) {
        if data.first == UInt8(ascii: "{") {
            guard let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            switch msg["op"] as? String {
            case "info" where (msg["kind"] as? String) == "depth":
                if connection == nil {  // first module to answer wins; drop the other probes
                    connection = c
                    for (h, p) in probes where h != host { p.cancel() }
                    probes = probes.filter { $0.key == host }
                }
                lastSeen = CACurrentMediaTime()
                width = msg["width"] as? Int ?? width
                height = msg["height"] as? Int ?? height
                if let f = msg["fov_deg"] as? [Double], f.count == 2 { fovDeg = (f[0], f[1]) }
                let s = msg["sensor"] as? String ?? "?"
                DispatchQueue.main.async { self.sensor = s; self.host = host }
            case "pong":
                if let t1 = msg["t1"] as? Double, let t2 = msg["t2"] as? Double {
                    let t3 = CACurrentMediaTime()
                    sync.append((t3 - t1, t2 - (t1 + t3) / 2))
                    if sync.count > 30 { sync.removeFirst(sync.count - 30) }
                }
            default:
                break
            }
            return
        }
        guard c === connection, let frame = assemble(data), let offset = clockOffset else { return }
        lastSeen = CACurrentMediaTime()
        counted += 1
        onFrame?(ModuleDepthFrame(t: frame.t - offset, width: frame.width, height: frame.height, mm: frame.mm))
    }

    /// "CSD1" | u8 3 | u8 1 | u16 w | u16 h | u32 frame | u16 chunk | u16 n | f64 t | u16 pixels...
    private func assemble(_ data: Data) -> ModuleDepthFrame? {
        let header = 26
        guard data.count > header, data.prefix(4) == Data("CSD1".utf8) else { return nil }
        let (w, h, fid, k, n, t) = data.withUnsafeBytes { raw in
            (Int(raw.loadUnaligned(fromByteOffset: 6, as: UInt16.self)),
             Int(raw.loadUnaligned(fromByteOffset: 8, as: UInt16.self)),
             raw.loadUnaligned(fromByteOffset: 10, as: UInt32.self),
             Int(raw.loadUnaligned(fromByteOffset: 14, as: UInt16.self)),
             Int(raw.loadUnaligned(fromByteOffset: 16, as: UInt16.self)),
             raw.loadUnaligned(fromByteOffset: 18, as: Double.self))
        }
        var f = parts[fid] ?? (n: n, t: t, w: w, h: h, chunks: [:])
        f.chunks[k] = data.subdata(in: header..<data.count)
        parts[fid] = f
        if parts.count > 4, let oldest = parts.keys.min(), oldest != fid { parts[oldest] = nil }
        guard f.chunks.count == f.n else { return nil }
        parts = parts.filter { $0.key > fid }
        var bytes = Data()
        for i in 0..<f.n { if let c = f.chunks[i] { bytes.append(c) } }
        guard bytes.count == 2 * w * h else { return nil }
        let mm = bytes.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }  // little-endian
        return ModuleDepthFrame(t: t, width: w, height: h, mm: mm)
    }
}

/// Writes module frames next to a session: tof/000000.png (16-bit mm), tof.jsonl and
/// depth_module.json (docs/capture_format.md).
final class DepthRecorder {
    private let directory: URL
    private let queue = DispatchQueue(label: "cleanscan.depthrecorder")
    private let lines: FileHandle?
    private var index = 0

    init(directory: URL) {
        self.directory = directory
        try? FileManager.default.createDirectory(at: directory.appendingPathComponent("tof"),
                                                 withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("tof.jsonl")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        lines = try? FileHandle(forWritingTo: url)
    }

    func add(_ f: ModuleDepthFrame, fovDeg: (Double, Double)) {
        queue.async {
            let name = String(format: "%06d", self.index)
            guard self.writePNG16(f.mm, width: f.width, height: f.height,
                                  to: self.directory.appendingPathComponent("tof/\(name).png")) else { return }
            let fx = Double(f.width) / 2 / tan(fovDeg.0 * .pi / 360)
            let fy = Double(f.height) / 2 / tan(fovDeg.1 * .pi / 360)
            let rec: [String: Any] = ["index": self.index, "t": f.t, "depth": "tof/\(name).png",
                                      "intrinsics": [fx, fy, Double(f.width) / 2, Double(f.height) / 2],
                                      "size": [f.width, f.height]]
            if var line = try? JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys]) {
                line.append(0x0A)
                try? self.lines?.write(contentsOf: line)
            }
            self.index += 1
        }
    }

    func finish(sensor: String, fovDeg: (Double, Double), rttMs: Double?) {
        queue.async {
            try? self.lines?.close()
            let meta: [String: Any] = ["sensor": sensor, "fov_deg": [fovDeg.0, fovDeg.1], "depth_unit": 0.001,
                                       "frame": "opencv", "clock_rtt_ms": rttMs ?? -1, "frames": self.index]
            if let d = try? JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys]) {
                try? d.write(to: self.directory.appendingPathComponent("depth_module.json"))
            }
        }
    }

    private func writePNG16(_ px: [UInt16], width: Int, height: Int, to url: URL) -> Bool {
        let data = px.withUnsafeBufferPointer { Data(buffer: $0) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        guard let provider = CGDataProvider(data: data as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: 16, bitsPerPixel: 16,
                                bytesPerRow: width * 2, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: info,
                                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { return false }
        CGImageDestinationAddImage(dest, img, nil)
        return CGImageDestinationFinalize(dest)
    }
}
