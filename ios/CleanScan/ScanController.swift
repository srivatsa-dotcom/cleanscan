import ARKit
import Combine
import RealityKit
import UIKit

/// Drives the AR session and records RGB + LiDAR depth + pose + people/animal masks.
final class ScanController: NSObject, ObservableObject, ARSessionDelegate {
    @Published var isRecording = false
    @Published var frameCount = 0
    @Published var maskedPercent: Double = 0
    @Published var status = "Point at the room, then press Record"
    @Published var maskPreview: UIImage?
    @Published var showMesh = true { didSet { updateDebugOptions() } }
    let lidarAvailable = ARWorldTrackingConfiguration.supportsFrameSemantics(.sceneDepth)

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
    }

    func startSession() {
        guard let view = arView else { return }
        let config = ARWorldTrackingConfiguration()
        config.worldAlignment = .gravity
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
        if !lidarAvailable {
            status = "This device has no LiDAR - scanning needs an iPhone/iPad Pro"
        }
    }

    private func updateDebugOptions() {
        guard let view = arView else { return }
        if showMesh { view.debugOptions.insert(.showSceneUnderstanding) } else { view.debugOptions.remove(.showSceneUnderstanding) }
    }

    // MARK: recording

    func toggleRecording() {
        isRecording ? stop() : start()
    }

    private func start() {
        guard lidarAvailable else { return }
        do {
            writer = try SessionWriter.create()
            frameCount = 0
            lastPose = nil
            isRecording = true
            status = "Recording - move slowly, cover every wall and the floor"
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
        queue.async {
            w?.finish()
            DispatchQueue.main.async { self.status = "Saved \(self.frameCount) frames. Open Scans to export." }
        }
    }

    // MARK: ARSessionDelegate

    func session(_ session: ARSession, didUpdate frame: ARFrame) {
        guard isRecording, !busy, let writer else { return }
        guard case .normal = frame.camera.trackingState else {
            status = "Tracking limited - slow down / add light"
            return
        }
        guard let depth = frame.sceneDepth else { return }
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
            depth: depth.depthMap,
            confidence: depth.confidenceMap,
            personSegmentation: frame.segmentationBuffer,
            intrinsics: frame.camera.intrinsics,
            imageResolution: frame.camera.imageResolution,
            cameraToWorld: pose)

        queue.async { [weak self] in
            guard let self else { return }
            let mask = self.masker.mask(for: item)
            let result = Result { try writer.write(item, mask: mask) }
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
                        : "Recording - move slowly, cover every wall and the floor"
                case .failure(let error):
                    self.status = "Write failed: \(error.localizedDescription)"
                    self.stop()
                }
            }
        }
    }

    func session(_ session: ARSession, didFailWithError error: Error) {
        status = "AR error: \(error.localizedDescription)"
    }
}

/// Everything recorded for one frame (pixel buffers are retained, the ARFrame is not).
struct CapturedFrame {
    let timestamp: TimeInterval
    let image: CVPixelBuffer
    let depth: CVPixelBuffer
    let confidence: CVPixelBuffer?
    let personSegmentation: CVPixelBuffer?
    let intrinsics: simd_float3x3
    let imageResolution: CGSize
    let cameraToWorld: simd_float4x4

    var depthSize: (width: Int, height: Int) {
        (CVPixelBufferGetWidth(depth), CVPixelBufferGetHeight(depth))
    }
}
