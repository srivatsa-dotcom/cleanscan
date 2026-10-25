import Foundation
import Network
import QuartzCore

/// One LiDAR reading on the phone's clock (same clock as ARFrame.timestamp).
struct RigReading {
    let t: Double
    let angle: Float
    let range: Float
    let intensity: UInt8
}

/// Talks to the CleanScan rig (Raspberry Pi + LiDAR, see rig/cleanscan_rig/protocol.py):
/// subscribes, keeps the Pi's clock in sync with the phone's, and delivers readings.
final class RigClient: ObservableObject {
    @Published private(set) var connected = false
    @Published private(set) var sensor = ""
    @Published private(set) var kind = ""
    @Published private(set) var pointsPerSecond = 0
    @Published private(set) var rttMs: Double?

    /// Called on the rig queue for every received batch (already on the phone clock).
    var onReadings: (([RigReading]) -> Void)?

    static let port: UInt16 = 5600
    private let queue = DispatchQueue(label: "cleanscan.rig")
    private var connection: NWConnection?
    private var timer: DispatchSourceTimer?
    private var sync: [(rtt: Double, offset: Double)] = []  // offset = rig clock - phone clock
    private var pingId = 0
    private var lastInfo = 0.0
    private var counted = 0
    private var countStart = CACurrentMediaTime()

    var clockOffset: Double? { sync.min(by: { $0.rtt < $1.rtt })?.offset }

    func start(host: String) {
        stop()
        let c = NWConnection(host: NWEndpoint.Host(host), port: NWEndpoint.Port(rawValue: Self.port)!, using: .udp)
        connection = c
        c.start(queue: queue)
        receive(on: c)
        let t = DispatchSource.makeTimerSource(queue: queue)
        t.schedule(deadline: .now(), repeating: 0.5)
        t.setEventHandler { [weak self] in self?.tick() }
        t.resume()
        timer = t
    }

    func stop() {
        timer?.cancel()
        timer = nil
        connection?.cancel()
        connection = nil
        sync.removeAll()
        DispatchQueue.main.async { self.connected = false; self.pointsPerSecond = 0; self.rttMs = nil }
    }

    // MARK: protocol

    private func tick() {
        send(["op": "hello"])
        pingId += 1
        send(["op": "ping", "id": pingId, "t1": CACurrentMediaTime()])
        let now = CACurrentMediaTime()
        let rate = Int(Double(counted) / max(now - countStart, 1e-3))
        counted = 0
        countStart = now
        let alive = now - lastInfo < 3
        let rtt = sync.min(by: { $0.rtt < $1.rtt }).map { $0.rtt * 1000 }
        DispatchQueue.main.async {
            self.pointsPerSecond = rate
            self.connected = alive
            self.rttMs = rtt
        }
    }

    private func send(_ obj: [String: Any]) {
        guard let c = connection, let data = try? JSONSerialization.data(withJSONObject: obj) else { return }
        c.send(content: data, completion: .contentProcessed { _ in })
    }

    private func receive(on c: NWConnection) {
        c.receiveMessage { [weak self] data, _, _, error in
            guard let self, self.connection === c else { return }
            if let data, !data.isEmpty { self.handle(data) }
            if error == nil { self.receive(on: c) }
        }
    }

    private func handle(_ data: Data) {
        if data.first == UInt8(ascii: "{") {
            guard let msg = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
            switch msg["op"] as? String {
            case "info":
                lastInfo = CACurrentMediaTime()
                let s = msg["sensor"] as? String ?? "?", k = msg["kind"] as? String ?? "?"
                DispatchQueue.main.async { self.sensor = s; self.kind = k }
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
        guard let readings = Self.decode(data), let offset = clockOffset else { return }
        lastInfo = CACurrentMediaTime()
        counted += readings.count
        let phone = readings.map { RigReading(t: $0.t - offset, angle: $0.angle, range: $0.range, intensity: $0.intensity) }
        onReadings?(phone)
    }

    /// "CSL1" | u8 kind | u8 0 | u16 count | f64 t_base | count x (f32 dt, f32 angle, f32 range, u8 intensity)
    static func decode(_ data: Data) -> [RigReading]? {
        let header = 16, point = 13
        guard data.count >= header, data.prefix(4) == Data("CSL1".utf8) else { return nil }
        return data.withUnsafeBytes { raw -> [RigReading]? in
            let count = Int(raw.loadUnaligned(fromByteOffset: 6, as: UInt16.self))
            guard raw.count == header + count * point else { return nil }
            let tBase = raw.loadUnaligned(fromByteOffset: 8, as: Double.self)
            var out: [RigReading] = []
            out.reserveCapacity(count)
            for k in 0..<count {
                let o = header + k * point
                out.append(RigReading(
                    t: tBase + Double(raw.loadUnaligned(fromByteOffset: o, as: Float.self)),
                    angle: raw.loadUnaligned(fromByteOffset: o + 4, as: Float.self),
                    range: raw.loadUnaligned(fromByteOffset: o + 8, as: Float.self),
                    intensity: raw.load(fromByteOffset: o + 12, as: UInt8.self)))
            }
            return out
        }
    }
}
