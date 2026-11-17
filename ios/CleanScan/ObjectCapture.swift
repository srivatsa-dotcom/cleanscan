import Combine
import CoreGraphics
import Foundation
import simd

// Object selection and guided capture for RL scenes (docs/RL_SCENES_PLAN.md).
// Same logic as android/.../ObjectCapture.kt, which is unit-tested on the JVM
// (tools/app-logic-test); keep the two in sync.

/// COCO class names in model order (YOLO-seg models trained on COCO).
let cocoNames: [String] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

/// How an object becomes interactive in the simulation.
enum CaptureMode: String, CaseIterable, Identifiable {
    case pickable, articulated, transparent
    var id: String { rawValue }

    var stages: [String] {
        switch self {
        case .pickable: return ["scan"]
        case .articulated: return ["closed", "open"]
        case .transparent: return ["profile", "top"]
        }
    }

    var label: String {
        switch self {
        case .pickable: return "Pick up"
        case .articulated: return "Opens"
        case .transparent: return "Glass"
        }
    }

    /// Same table as the desktop pipeline (pipeline/cleanscan/objects.py).
    static func forClass(_ cls: String) -> CaptureMode {
        let c = cls.trimmingCharacters(in: .whitespaces).lowercased()
        if ["refrigerator", "fridge", "cupboard", "cabinet", "drawer", "oven", "microwave", "door",
            "dishwasher", "toilet"].contains(c) { return .articulated }
        if ["wine glass", "glass", "tumbler", "steel tumbler"].contains(c) { return .transparent }
        return .pickable
    }
}

/// Classes offered as candidate objects (not living beings, not room furniture/vehicles).
let objectClasses: Set<Int> = {
    let excluded: Set<String> = ["bench", "chair", "couch", "bed", "dining table", "tv", "potted plant", "car", "bus",
                                 "train", "truck", "airplane", "boat", "bicycle", "motorcycle", "traffic light",
                                 "fire hydrant", "stop sign", "parking meter"]
    return Set((0..<cocoNames.count).filter { !LivingBeingMasker.livingClasses.contains($0) && !excluded.contains(cocoNames[$0]) })
}()

final class Track {
    let id: Int
    let classId: Int
    let firstFrame: Int
    var frames = 0
    var lastBox = CGRect.zero
    var lastSeen: Double = 0
    var bestScore: Float = 0
    var bestFrame = -1
    var className: String { classId < cocoNames.count ? cocoNames[classId] : "object" }

    init(id: Int, classId: Int, firstFrame: Int) {
        self.id = id
        self.classId = classId
        self.firstFrame = firstFrame
    }
}

private func iou(_ a: CGRect, _ b: CGRect) -> Float {
    let i = a.intersection(b)
    if i.isNull || i.isEmpty { return 0 }
    let inter = i.width * i.height
    return Float(inter / (a.width * a.height + b.width * b.height - inter))
}

/// Keeps object identities across frames (same class, best box overlap, seen recently).
final class ObjectTracker {
    private let iouThreshold: Float
    private let maxGap: Double
    private var nextId: Int
    private(set) var tracks: [Int: Track] = [:]
    private var order: [Int] = []

    init(iouThreshold: Float = 0.25, maxGapSeconds: Double = 1.5, firstId: Int = 1) {
        self.iouThreshold = iouThreshold
        maxGap = maxGapSeconds
        nextId = firstId
    }

    /// Track id for every detection, in the same order. Boxes are portrait-normalised.
    func update(frameIndex: Int, time: Double, classes: [Int], scores: [Float], boxes: [CGRect]) -> [Int] {
        var ids = [Int](repeating: -1, count: boxes.count)
        var taken = Set<Int>()
        for i in boxes.indices.sorted(by: { scores[$0] > scores[$1] }) {
            var best: Track?
            var bestIou = iouThreshold
            for t in tracks.values where t.classId == classes[i] && !taken.contains(t.id) && time - t.lastSeen <= maxGap {
                let v = iou(t.lastBox, boxes[i])
                if v > bestIou { bestIou = v; best = t }
            }
            let t: Track
            if let best { t = best } else {
                t = Track(id: nextId, classId: classes[i], firstFrame: frameIndex)
                tracks[nextId] = t
                order.append(nextId)
                nextId += 1
            }
            taken.insert(t.id)
            t.frames += 1
            t.lastBox = boxes[i]
            t.lastSeen = time
            let b = boxes[i]
            let inside = b.minX > 0.02 && b.minY > 0.02 && b.maxX < 0.98 && b.maxY < 0.98
            let s = scores[i] * (inside ? 1 : 0.7)
            if s > t.bestScore { t.bestScore = s; t.bestFrame = frameIndex }
            ids[i] = t.id
        }
        return ids
    }

    func confirmed(minFrames: Int = 3) -> [Track] {
        order.compactMap { tracks[$0] }.filter { $0.frames >= minFrames }
    }
}

/// Fraction of viewing directions (azimuth bins) covered around an object.
struct CoverageRing {
    let bins = 36
    private(set) var hit = [Bool](repeating: false, count: 36)

    /// Camera and object positions in the AR world (Y up).
    mutating func add(camera: SIMD3<Float>, object: SIMD3<Float>) {
        let a = atan2(Double(camera.z - object.z), Double(camera.x - object.x))
        let b = min(bins - 1, max(0, Int((a + .pi) / (2 * .pi) * Double(bins))))
        hit[b] = true
    }

    var fraction: Float { Float(hit.filter { $0 }.count) / Float(bins) }
}

/// Camera pitch in degrees (0 = level, -90 = straight down); the AR camera looks along -Z.
func cameraPitchDegrees(_ cameraToWorld: simd_float4x4) -> Float {
    let forwardY = -cameraToWorld.columns.2.y
    return asin(max(-1, min(1, forwardY))) * 180 / .pi
}

/// Portrait tap (0..1, top-left origin) -> sensor-image (landscape) coordinates.
func portraitToSensor(_ p: CGPoint) -> CGPoint { CGPoint(x: p.y, y: 1 - p.x) }

/// What the user chose on the selection screen.
final class SelectedObject: Identifiable, ObservableObject {
    let id: Int
    let cls: String
    let name: String
    @Published var mode: CaptureMode
    @Published var checked: Bool
    let source: String  // detected | tap
    let seed: (frame: Int, u: Double, v: Double)?
    let thumbnail: URL?
    let frames: Int
    var captures: [String: String] = [:]

    init(id: Int, cls: String, name: String, mode: CaptureMode, source: String, checked: Bool = false,
         seed: (frame: Int, u: Double, v: Double)? = nil, thumbnail: URL? = nil, frames: Int = 0) {
        self.id = id
        self.cls = cls
        self.name = name
        self.mode = mode
        self.source = source
        self.checked = checked
        self.seed = seed
        self.thumbnail = thumbnail
        self.frames = frames
    }
}

/// tracks.json / selection.json (docs/capture_format.md).
enum ObjectJSON {
    static func tracks(_ tracks: [Track]) -> Data {
        let arr: [[String: Any]] = tracks.map {
            ["id": $0.id, "class": $0.className, "coco_id": $0.classId, "score": Double($0.bestScore),
             "frames": $0.frames, "first_frame": $0.firstFrame, "best_frame": $0.bestFrame,
             "thumb": "thumbs/\($0.id).jpg"]
        }
        return (try? JSONSerialization.data(withJSONObject: arr, options: [.prettyPrinted, .sortedKeys])) ?? Data()
    }

    static func selection(_ objects: [SelectedObject]) -> Data {
        let arr: [[String: Any]] = objects.map { o in
            var d: [String: Any] = ["id": o.id, "class": o.cls, "name": o.name, "mode": o.mode.rawValue,
                                    "source": o.source, "captures": o.captures]
            if let s = o.seed { d["seed"] = ["frame": s.frame, "u": s.u, "v": s.v] }
            return d
        }
        return (try? JSONSerialization.data(withJSONObject: ["objects": arr], options: [.prettyPrinted, .sortedKeys])) ?? Data()
    }
}
