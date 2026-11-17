import ARKit
import CoreML
import UIKit
import Vision

/// Per-frame mask of living beings (people + animals) at depth-map resolution.
struct LivingMask {
    let width: Int
    let height: Int
    var pixels: [UInt8]  // 255 = person/animal, 0 = static scene

    var maskedFraction: Double {
        Double(pixels.reduce(0) { $0 + ($1 > 0 ? 1 : 0) }) / Double(max(pixels.count, 1))
    }

    /// Red overlay rotated to portrait for the on-screen preview.
    func previewImage() -> UIImage? {
        var rgba = [UInt8](repeating: 0, count: width * height * 4)
        for i in 0..<(width * height) where pixels[i] > 0 {
            rgba[i * 4] = 255
            rgba[i * 4 + 3] = 160
        }
        guard let provider = CGDataProvider(data: Data(rgba) as CFData),
              let cg = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                               bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                               provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent)
        else { return nil }
        return UIImage(cgImage: cg, scale: 1, orientation: .right)
    }
}

/// Combines three detectors so nothing alive ends up in the map:
///  1. ARKit people segmentation (pixel accurate, people only)
///  2. YOLO-seg Core ML model "LivingSeg" (people + 10 animal classes), if bundled
///  3. Fallback without the model: Vision animal detector (cats/dogs), box-masked
final class LivingBeingMasker {
    /// COCO ids: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
    static let livingClasses: Set<Int> = [0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
    private let confidence: Float = 0.30
    private let model: VNCoreMLModel?

    var hasModel: Bool { model != nil }

    init() {
        if let url = Bundle.main.url(forResource: "LivingSeg", withExtension: "mlmodelc"),
           let ml = try? MLModel(contentsOf: url, configuration: MLModelConfiguration()) {
            model = try? VNCoreMLModel(for: ml)
        } else {
            model = nil
        }
    }

    func mask(for f: CapturedFrame) -> LivingMask {
        analyze(f, detectObjects: false).mask
    }

    /// The living-being mask, plus (with ``detectObjects``) the other COCO objects the same
    /// YOLO pass found - portrait-normalised boxes and masks - for object selection.
    func analyze(_ f: CapturedFrame, detectObjects: Bool) -> (mask: LivingMask, objects: [YOLOSegDecoder.Detection]) {
        let (w, h) = f.depthSize
        var out = LivingMask(width: w, height: h, pixels: [UInt8](repeating: 0, count: w * h))
        if let seg = f.personSegmentation { addARKitPeople(seg, into: &out) }
        var objects: [YOLOSegDecoder.Detection] = []
        if let model {
            objects = addYOLO(model: model, image: f.image, into: &out, detectObjects: detectObjects)
        } else {
            addVisionAnimals(image: f.image, into: &out)
        }
        return (out, objects)
    }

    /// Detections -> 16-bit labels (track ids) over the landscape depth-resolution image.
    static func instanceLabels(_ dets: [YOLOSegDecoder.Detection], ids: [Int], width: Int, height: Int) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: width * height)
        guard !dets.isEmpty else { return out }
        for y in 0..<height {
            let xp = 1 - (Float(y) + 0.5) / Float(height)
            for x in 0..<width {
                let yp = (Float(x) + 0.5) / Float(width)
                for (k, d) in dets.enumerated() where ids[k] > 0 && d.contains(xp: xp, yp: yp) {
                    out[y * width + x] = UInt16(clamping: ids[k])
                    break
                }
            }
        }
        return out
    }

    /// One detection -> 0/255 mask over the landscape depth-resolution image.
    static func objectMask(_ d: YOLOSegDecoder.Detection?, width: Int, height: Int) -> [UInt8] {
        var out = [UInt8](repeating: 0, count: width * height)
        guard let d else { return out }
        for y in 0..<height {
            let xp = 1 - (Float(y) + 0.5) / Float(height)
            for x in 0..<width where d.contains(xp: xp, yp: (Float(x) + 0.5) / Float(width)) {
                out[y * width + x] = 255
            }
        }
        return out
    }

    // MARK: 1. ARKit people segmentation

    private func addARKitPeople(_ seg: CVPixelBuffer, into m: inout LivingMask) {
        CVPixelBufferLockBaseAddress(seg, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(seg, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(seg) else { return }
        let sw = CVPixelBufferGetWidth(seg), sh = CVPixelBufferGetHeight(seg)
        let stride = CVPixelBufferGetBytesPerRow(seg)
        let p = base.assumingMemoryBound(to: UInt8.self)
        for y in 0..<m.height {
            let sy = min(sh - 1, y * sh / m.height)
            for x in 0..<m.width {
                let sx = min(sw - 1, x * sw / m.width)
                if p[sy * stride + sx] > 0 { m.pixels[y * m.width + x] = 255 }
            }
        }
    }

    // MARK: 2. YOLO-seg (Ultralytics export, nms=False)

    /// The camera buffer is landscape; the phone is held in portrait, so the image is
    /// given to Vision rotated (.right) - detectors work much better on upright people.
    /// Portrait-normalised (xp, yp) <-> landscape-normalised (xl, yl): xl = yp, yl = 1 - xp.
    @discardableResult
    private func addYOLO(model: VNCoreMLModel, image: CVPixelBuffer, into m: inout LivingMask,
                         detectObjects: Bool = false) -> [YOLOSegDecoder.Detection] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .scaleFill
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .right)
        guard (try? handler.perform([request])) != nil,
              let obs = request.results as? [VNCoreMLFeatureValueObservation] else { return [] }
        let arrays = obs.compactMap { $0.featureValue.multiArrayValue }
        guard let det = arrays.first(where: { $0.shape.count == 3 }),
              let proto = arrays.first(where: { $0.shape.count == 4 }) else { return [] }
        let classes = detectObjects ? Self.livingClasses.union(objectClasses) : Self.livingClasses
        let all = YOLOSegDecoder.decode(det: det, proto: proto, classes: classes, confidence: confidence)
        let detections = all.filter { Self.livingClasses.contains($0.classId) }
        for d in detections {
            for y in 0..<m.height {
                let xp = 1 - (Float(y) + 0.5) / Float(m.height)
                for x in 0..<m.width {
                    let yp = (Float(x) + 0.5) / Float(m.width)
                    if d.contains(xp: xp, yp: yp) { m.pixels[y * m.width + x] = 255 }
                }
            }
        }
        return all.filter { !Self.livingClasses.contains($0.classId) }
    }

    // MARK: 3. Fallback: Vision animal boxes

    private func addVisionAnimals(image: CVPixelBuffer, into m: inout LivingMask) {
        let request = VNRecognizeAnimalsRequest()
        let handler = VNImageRequestHandler(cvPixelBuffer: image, orientation: .right)
        guard (try? handler.perform([request])) != nil, let results = request.results else { return }
        for r in results where r.confidence > confidence {
            // Vision boxes: normalised, origin bottom-left, in the oriented (portrait) image.
            let b = r.boundingBox.insetBy(dx: -0.05 * r.boundingBox.width, dy: -0.05 * r.boundingBox.height)
            for y in 0..<m.height {
                let xp = 1 - (CGFloat(y) + 0.5) / CGFloat(m.height)
                for x in 0..<m.width {
                    let ypTop = (CGFloat(x) + 0.5) / CGFloat(m.width)
                    if b.contains(CGPoint(x: xp, y: 1 - ypTop)) { m.pixels[y * m.width + x] = 255 }
                }
            }
        }
    }
}
