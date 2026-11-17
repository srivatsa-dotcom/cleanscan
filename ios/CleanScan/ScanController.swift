import ARKit
import Combine
import CoreImage
import ImageIO
import RealityKit
import UIKit

/// Drives the AR session and records RGB + LiDAR depth + pose + people/animal masks.
/// Without LiDAR (e.g. iPhone 11) it records camera-only scans; depth is then computed
/// on the computer from stereo between frames (pipeline/cleanscan/stereo.py).
final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isRecording = false
    @Published var frameCount = 0
    @Published var maskedPercent: Double = 0
    @Published var status = "Point at the room, then press Record"
    @Published var maskPreview: UIImage?
    @Published var showMesh = true { didSet { updateDebugOptions() } }
    let lidarAvailable = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

    /// External LiDAR rig (Raspberry Pi + LD06/TF-Luna on the phone, see rig/).
    let rig = RigClient()
    @Published var rigEnabled = UserDefaults.standard.bool(forKey: "rigEnabled") {
        didSet { UserDefaults.standard.set(rigEnabled, forKey: "rigEnabled"); updateRig() }
    }
    @Published var rigHost = UserDefaults.standard.string(forKey: "rigHost") ?? "10.42.0.1" {
        didSet { UserDefaults.standard.set(rigHost, forKey: "rigHost"); updateRig() }
    }
    private var streams: StreamRecorder?

    /// Clip-on depth module (module/): ToF frames recorded next to the camera frames.
    let depthModule = DepthModuleClient()
    @Published var depthModuleEnabled = UserDefaults.standard.bool(forKey: "depthModuleEnabled") {
        didSet { UserDefaults.standard.set(depthModuleEnabled, forKey: "depthModuleEnabled"); updateDepthModule() }
    }
    @Published var depthModuleHost = UserDefaults.standard.string(forKey: "depthModuleHost") ?? "192.168.4.1" {
        didSet { UserDefaults.standard.set(depthModuleHost, forKey: "depthModuleHost"); updateDepthModule() }
    }
    private var depthRecorder: DepthRecorder?

    // MARK: RL scenes: object selection + guided capture (docs/RL_SCENES_PLAN.md)
    enum Phase { case idle, room, selecting, capturing }
    struct Keyframe: Identifiable { let id: Int; let image: UIImage }
    @Published var phase: Phase = .idle
    @Published var candidates: [SelectedObject] = []
    @Published var keyframes: [Keyframe] = []
    @Published var guideTitle = ""
    @Published var guideText = ""
    @Published var guideMode: CaptureMode = .pickable
    @Published var guideStage = ""
    @Published var coverage: Float = 0
    @Published var pitch: Float = 0
    private var tracker: ObjectTracker?  // room pass, used on `queue` only
    private var scanURL: URL?
    private var selected: [SelectedObject] = []
    private var current = 0
    private var stage = 0
    private var nextTapId = 101
    private var ring: CoverageRing?
    private var objectCenter: SIMD3<Float>?
    private var captureClass: Int?  // nil: no object masks, -1: most central detection
    private var gate: ((simd_float4x4) -> Bool)?
    private var lastGuideUpdate: TimeInterval = 0
    private let ciContext = CIContext()

    private weak var arView: ARView?
    private var writer: SessionWriter?
    private let masker = LivingBeingMasker()
    private let queue = DispatchQueue(label: "cleanscan.capture", qos: .userInitiated)
    private var busy = false  // only main thread touches this
    private var lastPose: simd_float4x4?
    private var lastCapture: TimeInterval = 0

    // capture policy
    private let minInterval: TimeInterval = 0.2  // <= 5 fps
    private let maxInterval: TimeInterval = 0.5  // still record while standing (people walk away)
    private let minTranslation: Float = 0.01
    private let minRotationDeg: Float = 2

    var hasSegmentationModel: Bool { masker.hasModel }

    func attach(_ view: ARView) {
        arView = view
        view.session.delegate = self
        view.renderOptions.insert(.disableMotionBlur)
        startSession()
        updateRig()
        updateDepthModule()
    }

    private func updateRig() {
        if rigEnabled { rig.start(host: rigHost) } else { rig.stop() }
    }

    private func updateDepthModule() {
        if depthModuleEnabled { depthModule.start(preferredHost: depthModuleHost) } else { depthModule.stop() }
    }

    /// Record the module's frames into this session folder (room pass or a guided capture).
    private func startDepthRecording(in directory: URL) {
        guard depthModuleEnabled else { return }
        let rec = DepthRecorder(directory: directory)
        depthRecorder = rec
        let client = depthModule
        client.onFrame = { rec.add($0, fovDeg: client.fovDeg) }
    }

    private func finishDepthRecording() {
        depthModule.onFrame = nil
        depthRecorder?.finish(sensor: depthModule.sensor, fovDeg: depthModule.fovDeg, rttMs: depthModule.rttMs)
        depthRecorder = nil
    }

    var cameraOnly: Bool { !lidarAvailable }

    private var idleHint: String {
        cameraOnly
            ? "No LiDAR: camera-only mode. Walk slowly sideways along the walls, phone facing them"
            : "Point at the room, then press Record"
    }

    private var recordingHint: String {
        cameraOnly
            ? "Recording - keep walking sideways, slowly; avoid turning on the spot"
            : "Recording - move slowly, cover every wall and the floor"
    }

    func startSession() {
        guard let view = arView else { return }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
        config.planeDetection = [.horizontal, .vertical]  // saved with the scan for rig calibration
        if ARWorldTrackingConfiguration.supportsSceneReconstruction(.mesh) {
            config.sceneReconstruction = .mesh  // live preview only; the final map is fused offline
        }
        var semantics: ARConfiguration.FrameSemantics = []
        if lidarAvailable { semantics.insert(.sceneDepth) }
        if ARWorldTrackingConfiguration.supportsFrameSemantics(semantics.union(.personSegmentationWithDepth)) {
            semantics.insert(.personSegmentationWithDepth)
        }
        config.frameSemantics = semantics
        // Highest-resolution 4:3 video format keeps colour aligned with the depth map.
        if let fmt = ARWorldTrackingConfiguration.supportedVideoFormats
            .filter({ abs(Double($0.imageResolution.width / $0.imageResolution.height) - 4.0 / 3.0) < 0.01 })
            .max(by: { $0.imageResolution.width < $1.imageResolution.width }) {
            config.videoFormat = fmt
        }
        view.session.run(config, options: [.resetTracking, .removeExistingAnchors])
        updateDebugOptions()
        status = idleHint
    }

    private func updateDebugOptions() {
        guard let view = arView else { return }
        // LiDAR devices show the live mesh; others show ARKit's tracked feature points.
        let option: ARView.DebugOptions = cameraOnly ? .showFeaturePoints : .showSceneUnderstanding
        if showMesh { view.debugOptions.insert(option) } else { view.debugOptions.remove(option) }
    }

    // MARK: recording

    func toggleRecording() {
        switch phase {
        case .capturing: isRecording ? stopStage() : startStage()
        case .selecting: break
        default: isRecording ? stop() : start()
        }
    }

    private func start() {
        do {
            let w = try SessionWriter.create(depthSource: cameraOnly ? "none" : "lidar")
            writer = w
            let s = try StreamRecorder(directory: w.directory)
            streams = s
            rig.onReadings = { s.addLidar($0) }
            startDepthRecording(in: w.directory)
            frameCount = 0
            lastPose = nil
            tracker = ObjectTracker()
            keyframes = []
            candidates = []
            scanURL = w.directory
            phase = .room
            isRecording = true
            status = recordingHint
            UIApplication.shared.isIdleTimerDisabled = true
        } catch {
            status = "Cannot start: \(error.localizedDescription)"
        }
    }

    private func stop() {
        isRecording = false
        UIApplication.shared.isIdleTimerDisabled = false
        let w = writer
        writer = nil
        rig.onReadings = nil
        finishDepthRecording()
        let rigInfo: [String: Any]? = rigEnabled ? [
            "sensor": rig.sensor, "kind": rig.kind, "host": rigHost,
            "clock_rtt_ms": rig.rttMs ?? -1,
        ] : nil
        let planes = arView?.session.currentFrame?.anchors.compactMap { $0 as? ARPlaneAnchor } ?? []
        let s = streams
        streams = nil
        queue.async {
            w?.finish()
            // the detector's object tracks -> tracks.json and the selection screen
            let tracks = self.tracker?.confirmed() ?? []
            self.tracker = nil
            if let w { w.writeFile("tracks.json", data: ObjectJSON.tracks(tracks)) }
            let dir = w?.directory
            let cands = tracks.map { t in
                SelectedObject(id: t.id, cls: t.className, name: t.className, mode: CaptureMode.forClass(t.className),
                               source: "detected", thumbnail: dir?.appendingPathComponent("thumbs/\(t.id).jpg"),
                               frames: t.frames)
            }
            let done = {
                DispatchQueue.main.async {
                    self.candidates = cands
                    self.phase = .selecting
                    self.status = "Saved the room (\(self.frameCount) frames). Choose interactive objects."
                }
            }
            if let s { s.finish(rig: rigInfo, planes: planes, completion: done) } else { done() }
        }
    }

    // MARK: selection

    var selectionSummary: String {
        if candidates.filter({ $0.source == "detected" }).isEmpty {
            return "No objects were detected. Add objects by tapping them on a photo."
        }
        var counts: [String: Int] = [:]
        for c in candidates where c.source == "detected" { counts[c.cls, default: 0] += 1 }
        let parts = counts.sorted { $0.key < $1.key }.map { $0.value == 1 ? $0.key : "\($0.value)× \($0.key)" }
        return "Found: " + parts.joined(separator: ", ") + ". Which should be interactive for RL?"
    }

    /// An object the detector missed, tapped on a room-pass photo (portrait point 0..1).
    func addTappedObject(keyframe: Keyframe, at p: CGPoint, name: String, mode: CaptureMode) {
        let s = portraitToSensor(p)
        let label = name.trimmingCharacters(in: .whitespaces).isEmpty ? "object" : name
        candidates.append(SelectedObject(id: nextTapId, cls: label.lowercased(), name: label, mode: mode, source: "tap",
                                         checked: true, seed: (keyframe.id, Double(s.x), Double(s.y))))
        nextTapId += 1
    }

    func startCaptures() {
        selected = candidates.filter { $0.checked }
        current = 0
        stage = 0
        if selected.isEmpty { finishFlow() } else { phase = .capturing; showStage() }
    }

    func roomOnly() {
        selected = []
        finishFlow()
    }

    // MARK: guided capture

    private var currentObject: SelectedObject { selected[current] }
    private var stageName: String { currentObject.mode.stages[stage] }

    private func showStage() {
        let o = currentObject
        let n = o.name
        let k = "(\(current + 1)/\(selected.count))"
        guideMode = o.mode
        guideStage = stageName
        coverage = 0
        switch (o.mode, stageName) {
        case (.pickable, _):
            guideTitle = "Scan the \(n) \(k)"
            guideText = "Put it on a flat surface with space around it, point the phone at it, press Record and walk slowly all the way around it."
        case (.articulated, "closed"):
            guideTitle = "Scan the \(n) closed \(k)"
            guideText = "Close it. Stand 1.5–2 m away so all of it is in view, press Record and sweep slowly from side to side."
        case (.articulated, _):
            guideTitle = "Now open the \(n) fully"
            guideText = "Open the door / pull out the drawer as far as it goes. Press Record and scan it again the same way."
        case (.transparent, "profile"):
            guideTitle = "Side profile of the \(n) \(k)"
            guideText = "Hold the phone level at the height of the \(n), about 30 cm away. Press Record and move sideways around its front."
        default:
            guideTitle = "Top view of the \(n)"
            guideText = "Hold the phone flat above the \(n), screen up, so it is in the middle of the picture. Press Record for a second."
        }
        status = "Guided capture"
    }

    private func startStage() {
        guard let scanURL, phase == .capturing else { return }
        let o = currentObject
        let rel = "captures/\(o.id)/\(stageName)"
        do {
            let w = try SessionWriter.createCapture(scan: scanURL, relative: rel, depthSource: cameraOnly ? "none" : "lidar")
            writer = w
            let s = try StreamRecorder(directory: w.directory)
            streams = s
            rig.onReadings = { s.addLidar($0) }
            startDepthRecording(in: w.directory)
        } catch {
            status = "Cannot start: \(error.localizedDescription)"
            return
        }
        o.captures[stageName] = rel
        if o.mode == .articulated {
            captureClass = nil  // the pipeline finds the moving part itself
        } else {
            captureClass = cocoNames.firstIndex(of: o.cls) ?? -1
        }
        switch (o.mode, stageName) {
        case (.transparent, "profile"): gate = { abs(cameraPitchDegrees($0)) < 15 }
        case (.transparent, "top"): gate = { cameraPitchDegrees($0) < -65 }
        default: gate = nil
        }
        ring = o.mode == .pickable ? CoverageRing() : nil
        objectCenter = o.mode == .pickable ? pointedAt() : nil
        frameCount = 0
        lastPose = nil
        isRecording = true
    }

    /// What the user points at: a raycast hit at the screen centre, else 50 cm ahead.
    private func pointedAt() -> SIMD3<Float>? {
        guard let view = arView, let frame = view.session.currentFrame else { return nil }
        let c = CGPoint(x: view.bounds.midX, y: view.bounds.midY)
        if let hit = view.raycast(from: c, allowing: .estimatedPlane, alignment: .any).first {
            let p = hit.worldTransform.columns.3
            return SIMD3(p.x, p.y, p.z)
        }
        let t = frame.camera.transform
        let p = t.columns.3 - 0.5 * t.columns.2
        return SIMD3(p.x, p.y, p.z)
    }

    private func stopStage(advanceAfter: Bool = true) {
        isRecording = false
        let w = writer
        writer = nil
        rig.onReadings = nil
        finishDepthRecording()
        let s = streams
        streams = nil
        let cov = ring?.fraction
        ring = nil
        gate = nil
        captureClass = nil
        queue.async {
            w?.finish()
            s?.finish(rig: nil, planes: []) {}
        }
        if let cov, cov < 0.8 {
            status = "Only \(Int(cov * 100)) % of the way around was covered"
        }
        if advanceAfter { advance() }
    }

    /// Leave this object in the static room (its partial captures are not used).
    func skipObject() {
        if isRecording { stopStage(advanceAfter: false) }
        currentObject.captures = [:]
        stage = currentObject.mode.stages.count - 1
        advance()
    }

    private func advance() {
        stage += 1
        if stage >= currentObject.mode.stages.count {
            stage = 0
            current += 1
        }
        if current >= selected.count { finishFlow() } else { showStage() }
    }

    private func finishFlow() {
        let data = ObjectJSON.selection(selected)
        let url = scanURL
        let n = selected.count
        queue.async {
            if let url { try? data.write(to: url.appendingPathComponent("selection.json")) }
            DispatchQueue.main.async {
                self.phase = .idle
                self.status = n == 0 ? "Saved (room only). Open Scans to export."
                    : "Saved with \(n) interactive objects. Open Scans to export."
            }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        if phase == .capturing, frame.timestamp - lastGuideUpdate > 0.15 {
            lastGuideUpdate = frame.timestamp
            pitch = cameraPitchDegrees(frame.camera.transform)  // shown before and while recording
            if let r = ring { coverage = r.fraction }
        }
        guard isRecording else { return }
        guard case .normal = frame.camera.trackingState else {
            status = "Tracking limited - slow down / add light"
            return
        }
        streams?.addPose(t: frame.timestamp, cameraToWorld: frame.camera.transform)  // every frame (~60 Hz)
        if phase == .capturing {
            let t = frame.camera.transform
            if ring != nil, let c = objectCenter {
                ring?.add(camera: SIMD3(t.columns.3.x, t.columns.3.y, t.columns.3.z), object: c)
            }
            if let gate, !gate(t) { return }
        }
        guard !busy, let writer else { return }
        let depth = frame.sceneDepth
        if !cameraOnly && depth == nil { return }
        let now = frame.timestamp
        guard now - lastCapture >= minInterval else { return }
        let pose = frame.camera.transform
        if let last = lastPose, now - lastCapture < maxInterval {
            let dt = simd_distance(last.columns.3, pose.columns.3)
            let rel = simd_mul(simd_inverse(last), pose)
            let angle = acos(min(1, max(-1, (rel.columns.0.x + rel.columns.1.y + rel.columns.2.z - 1) / 2))) * 180 / .pi
            if dt < minTranslation && angle < minRotationDeg { return }
        }
        lastPose = pose
        lastCapture = now
        busy = true

        // Copy what we need and release the ARFrame immediately (ARKit has a small buffer pool).
        let item = CapturedFrame(
            timestamp: now,
            image: frame.capturedImage,
            depth: depth?.depthMap,
            confidence: depth?.confidenceMap,
            personSegmentation: frame.segmentationBuffer,
            intrinsics: frame.camera.intrinsics,
            imageResolution: frame.camera.imageResolution,
            cameraToWorld: pose)

        let cc = captureClass
        queue.async { [weak self] in
            guard let self else { return }
            let tracker = self.tracker
            let (mask, objects) = self.masker.analyze(item, detectObjects: tracker != nil || cc != nil)
            let (dw, dh) = item.depthSize
            var instances: [UInt16]?
            if let tracker {
                let index = writer.frameIndex
                let ids = tracker.update(frameIndex: index, time: item.timestamp, classes: objects.map { $0.classId },
                                         scores: objects.map { $0.score }, boxes: objects.map { $0.box })
                instances = LivingBeingMasker.instanceLabels(objects, ids: ids, width: dw, height: dh)
                for (k, d) in objects.enumerated() {
                    if let t = tracker.tracks[ids[k]], t.bestFrame == index, let jpg = self.thumbnail(item.image, d.box) {
                        writer.writeFile("thumbs/\(t.id).jpg", data: jpg)
                    }
                }
                if index % 10 == 0, let img = self.keyframeImage(item.image) {
                    DispatchQueue.main.async { if self.keyframes.count < 24 { self.keyframes.append(Keyframe(id: index, image: img)) } }
                }
            }
            var objectMask: [UInt8]?
            if let cc {
                let target = objects.filter { cc < 0 || $0.classId == cc }
                    .min { hypot($0.box.midX - 0.5, $0.box.midY - 0.5) < hypot($1.box.midX - 0.5, $1.box.midY - 0.5) }
                objectMask = LivingBeingMasker.objectMask(target, width: dw, height: dh)
            }
            let result = Result { try writer.write(item, mask: mask, instances: instances, objectMask: objectMask) }
            let preview = mask.previewImage()
            DispatchQueue.main.async {
                self.busy = false
                switch result {
                case .success:
                    self.frameCount += 1
                    self.maskedPercent = mask.maskedFraction * 100
                    self.maskPreview = preview
                    self.status = mask.maskedFraction > 0.001
                        ? "Recording - people/animals detected and removed"
                        : self.recordingHint
                case .failure(let error):
                    self.status = "Write failed: \(error.localizedDescription)"
                    self.stop()
                }
            }
        }
    }

    /// Crop of a detection (portrait-normalised box) from the landscape camera image, upright JPEG.
    private func thumbnail(_ buffer: CVPixelBuffer, _ box: CGRect) -> Data? {
        let img = CIImage(cvPixelBuffer: buffer)
        let W = img.extent.width, H = img.extent.height
        let b = box.insetBy(dx: -0.03, dy: -0.03)
        // portrait (xp, yp) -> landscape (xl = yp, yl = 1 - xp); CIImage has a bottom-left origin
        let rect = CGRect(x: b.minY * W, y: b.minX * H, width: b.height * W, height: b.width * H).intersection(img.extent)
        guard !rect.isEmpty else { return nil }
        let crop = img.cropped(to: rect).oriented(.right)
        return ciContext.jpegRepresentation(of: crop, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
                                            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.85])
    }

    /// Small upright copy of a room-pass frame for tapping objects the detector missed.
    private func keyframeImage(_ buffer: CVPixelBuffer) -> UIImage? {
        let img = CIImage(cvPixelBuffer: buffer).oriented(.right)
        let scale = 480 / img.extent.height
        let small = img.transformed(by: CGAffineTransform(scaleX: scale, y: scale))
        guard let cg = ciContext.createCGImage(small, from: small.extent) else { return nil }
        return UIImage(cgImage: cg)
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        status = "AR error: \(error.localizedDescription)"
    }
}

/// Everything recorded for one frame (pixel buffers are retained, the ARFrame is not).
struct CapturedFrame {
    let timestamp: TimeInterval
    let image: CVPixelBuffer
    let depth: CVPixelBuffer?  // nil in camera-only mode
    let confidence: CVPixelBuffer?
    let personSegmentation: CVPixelBuffer?
    let intrinsics: simd_float3x3
    let imageResolution: CGSize
    let cameraToWorld: simd_float4x4

    /// Resolution of depth, confidence and mask images (LiDAR depth size, else 256x192).
    var depthSize: (width: Int, height: Int) {
        guard let depth else { return (256, 192) }
        return (CVPixelBufferGetWidth(depth), CVPixelBufferGetHeight(depth))
    }
}
