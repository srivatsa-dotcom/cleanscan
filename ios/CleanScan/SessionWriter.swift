import ARKit
import CoreImage
import ImageIO
import UIKit
import UniformTypeIdentifiers

/// Writes a capture session in the CleanScan format (docs/capture_format.md).
final class SessionWriter {
    let directory: URL
    private let framesHandle: FileHandle
    private var index = 0
    private let ciContext = CIContext(options: [.useSoftwareRenderer: false])

    static var scansDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return docs.appendingPathComponent("Scans", isDirectory: true)
    }

    static func create() throws -> SessionWriter {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyyMMdd-HHmmss"
        let dir = scansDirectory.appendingPathComponent("Scan-\(fmt.string(from: Date()))", isDirectory: true)
        return try SessionWriter(directory: dir)
    }

    private init(directory: URL) throws {
        self.directory = directory
        let fm = FileManager.default
        for sub in ["color", "depth", "confidence", "mask"] {
            try fm.createDirectory(at: directory.appendingPathComponent(sub), withIntermediateDirectories: true)
        }
        let meta: [String: Any] = [
            "format": "cleanscan-capture",
            "version": 1,
            "platform": "ios",
            "device": UIDevice.current.model + " " + UIDevice.current.systemVersion,
            "depth_unit": 0.001,
            "confidence_levels": 3,  // ARConfidenceLevel low/medium/high
            "world_up": "y",
            "camera_convention": "opengl",
        ]
        try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
            .write(to: directory.appendingPathComponent("session.json"))
        let framesURL = directory.appendingPathComponent("frames.jsonl")
        fm.createFile(atPath: framesURL.path, contents: nil)
        framesHandle = try FileHandle(forWritingTo: framesURL)
    }

    func write(_ f: CapturedFrame, mask: LivingMask) throws {
        let name = String(format: "%06d", index)
        let (dw, dh) = f.depthSize

        // colour: full-resolution JPEG in sensor orientation (matches intrinsics)
        let colorURL = directory.appendingPathComponent("color/\(name).jpg")
        let ci = CIImage(cvPixelBuffer: f.image)
        guard let jpg = ciContext.jpegRepresentation(
            of: ci, colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!,
            options: [kCGImageDestinationLossyCompressionQuality as CIImageRepresentationOption: 0.9])
        else { throw CocoaError(.fileWriteUnknown) }
        try jpg.write(to: colorURL)

        // depth: metres (Float32) -> millimetres (UInt16 PNG)
        var depthMM = [UInt16](repeating: 0, count: dw * dh)
        CVPixelBufferLockBaseAddress(f.depth, .readOnly)
        if let base = CVPixelBufferGetBaseAddress(f.depth) {
            let stride = CVPixelBufferGetBytesPerRow(f.depth) / MemoryLayout<Float32>.size
            let p = base.assumingMemoryBound(to: Float32.self)
            for y in 0..<dh {
                for x in 0..<dw {
                    let d = p[y * stride + x]
                    depthMM[y * dw + x] = d.isFinite && d > 0 ? UInt16(min(65535, (d * 1000).rounded())) : 0
                }
            }
        }
        CVPixelBufferUnlockBaseAddress(f.depth, .readOnly)
        try writePNG16(depthMM, width: dw, height: dh, to: directory.appendingPathComponent("depth/\(name).png"))

        var confPath: String?
        if let conf = f.confidence {
            var c = [UInt8](repeating: 0, count: dw * dh)
            CVPixelBufferLockBaseAddress(conf, .readOnly)
            if let base = CVPixelBufferGetBaseAddress(conf) {
                let stride = CVPixelBufferGetBytesPerRow(conf)
                let p = base.assumingMemoryBound(to: UInt8.self)
                for y in 0..<dh { for x in 0..<dw { c[y * dw + x] = p[y * stride + x] } }
            }
            CVPixelBufferUnlockBaseAddress(conf, .readOnly)
            try writePNG8(c, width: dw, height: dh, to: directory.appendingPathComponent("confidence/\(name).png"))
            confPath = "confidence/\(name).png"
        }

        try writePNG8(mask.pixels, width: mask.width, height: mask.height,
                      to: directory.appendingPathComponent("mask/\(name).png"))

        // intrinsics are given for the captured image; scale them to the depth map
        let K = f.intrinsics
        let iw = Double(f.imageResolution.width), ih = Double(f.imageResolution.height)
        let sx = Double(dw) / iw, sy = Double(dh) / ih
        let fx = Double(K[0][0]), fy = Double(K[1][1]), cx = Double(K[2][0]), cy = Double(K[2][1])
        var pose: [Double] = []
        for r in 0..<4 { for c in 0..<4 { pose.append(Double(f.cameraToWorld[c][r])) } }  // row-major

        var rec: [String: Any] = [
            "index": index,
            "timestamp": f.timestamp,
            "color": "color/\(name).jpg",
            "depth": "depth/\(name).png",
            "mask": "mask/\(name).png",
            "color_intrinsics": [fx, fy, cx, cy],
            "color_size": [Int(iw), Int(ih)],
            "depth_intrinsics": [fx * sx, fy * sy, cx * sx, cy * sy],
            "depth_size": [dw, dh],
            "camera_to_world": pose,
        ]
        if let confPath { rec["confidence"] = confPath }
        var line = try JSONSerialization.data(withJSONObject: rec, options: [.sortedKeys])
        line.append(0x0A)
        try framesHandle.write(contentsOf: line)
        index += 1
    }

    func finish() {
        try? framesHandle.synchronize()
        try? framesHandle.close()
    }

    // MARK: PNG helpers

    private func writePNG16(_ px: [UInt16], width: Int, height: Int, to url: URL) throws {
        let data = px.withUnsafeBufferPointer { Data(buffer: $0) }
        let info = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue)
        try writePNG(data, width: width, height: height, bits: 16, info: info, to: url)
    }

    private func writePNG8(_ px: [UInt8], width: Int, height: Int, to url: URL) throws {
        try writePNG(Data(px), width: width, height: height, bits: 8,
                     info: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue), to: url)
    }

    private func writePNG(_ data: Data, width: Int, height: Int, bits: Int, info: CGBitmapInfo, to url: URL) throws {
        guard let provider = CGDataProvider(data: data as CFData),
              let img = CGImage(width: width, height: height, bitsPerComponent: bits, bitsPerPixel: bits,
                                bytesPerRow: width * bits / 8, space: CGColorSpaceCreateDeviceGray(),
                                bitmapInfo: info, provider: provider, decode: nil,
                                shouldInterpolate: false, intent: .defaultIntent),
              let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        CGImageDestinationAddImage(dest, img, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}

/// Lists, zips (for AirDrop / Files / email) and deletes recorded sessions.
enum ScanLibrary {
    static func list() -> [URL] {
        let dir = SessionWriter.scansDirectory
        let items = (try? FileManager.default.contentsOfDirectory(
            at: dir, includingPropertiesForKeys: [.creationDateKey], options: [.skipsHiddenFiles])) ?? []
        return items.filter { $0.hasDirectoryPath }.sorted { $0.lastPathComponent > $1.lastPathComponent }
    }

    static func frameCount(_ url: URL) -> Int {
        guard let s = try? String(contentsOf: url.appendingPathComponent("frames.jsonl"), encoding: .utf8) else { return 0 }
        return s.split(separator: "\n").count
    }

    /// Uses NSFileCoordinator's built-in directory zipping (no third-party code).
    static func zip(_ url: URL) throws -> URL {
        var result: URL?
        var coordError: NSError?
        var copyError: Error?
        NSFileCoordinator().coordinate(readingItemAt: url, options: [.forUploading], error: &coordError) { zipURL in
            let dest = FileManager.default.temporaryDirectory.appendingPathComponent(url.lastPathComponent + ".zip")
            try? FileManager.default.removeItem(at: dest)
            do { try FileManager.default.copyItem(at: zipURL, to: dest); result = dest } catch { copyError = error }
        }
        if let e = coordError ?? copyError { throw e }
        guard let result else { throw CocoaError(.fileWriteUnknown) }
        return result
    }

    static func delete(_ url: URL) {
        try? FileManager.default.removeItem(at: url)
    }
}
