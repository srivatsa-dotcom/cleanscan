import CoreGraphics
import CoreML

/// Decodes the raw outputs of an Ultralytics YOLO-seg model exported with nms=False:
///   det   [1, 4 + numClasses + 32, N]  (cx, cy, w, h in input pixels or 0..1, class scores, mask coeffs)
///   proto [1, 32, H/4, W/4]
struct YOLOSegDecoder {
    struct Detection {
        let classId: Int
        let score: Float
        let box: CGRect  // normalised [0,1] in the model input image (top-left origin)
        let mask: [Bool]  // maskWidth * maskHeight over the whole input image
        let maskWidth: Int
        let maskHeight: Int

        func contains(xp: Float, yp: Float) -> Bool {
            guard box.contains(CGPoint(x: CGFloat(xp), y: CGFloat(yp))) else { return false }
            let mx = min(maskWidth - 1, max(0, Int(xp * Float(maskWidth))))
            let my = min(maskHeight - 1, max(0, Int(yp * Float(maskHeight))))
            return mask[my * maskWidth + mx]
        }
    }

    static func decode(det: MLMultiArray, proto: MLMultiArray, classes: Set<Int>,
                       confidence: Float, iou: Float = 0.5, maxDetections: Int = 30) -> [Detection] {
        let C = det.shape[1].intValue, N = det.shape[2].intValue
        let nm = proto.shape[1].intValue, mh = proto.shape[2].intValue, mw = proto.shape[3].intValue
        let nc = C - 4 - nm
        guard nc > 0 else { return [] }
        let D = contiguous(det), P = contiguous(proto)
        let inW = Float(mw * 4), inH = Float(mh * 4)

        // Core ML exports emit boxes in input pixels, TFLite/LiteRT exports normalised to 0..1.
        // Anchor centres span the whole image, so the largest cx tells the two apart.
        let maxCx = (0..<N).reduce(Float(0)) { max($0, D[$1]) }
        let sx: Float = maxCx <= 2 ? 1 : inW, sy: Float = maxCx <= 2 ? 1 : inH

        struct Cand { var cls: Int; var score: Float; var box: CGRect; var coeffs: [Float] }
        var cands: [Cand] = []
        for i in 0..<N {
            var best: Float = 0, bestC = -1
            for c in classes where c < nc {
                let s = D[(4 + c) * N + i]
                if s > best { best = s; bestC = c }
            }
            guard best >= confidence, bestC >= 0 else { continue }
            let cx = D[i] / sx, cy = D[N + i] / sy, w = D[2 * N + i] / sx, h = D[3 * N + i] / sy
            let box = CGRect(x: CGFloat(cx - w / 2), y: CGFloat(cy - h / 2), width: CGFloat(w), height: CGFloat(h))
            let coeffs = (0..<nm).map { D[(4 + nc + $0) * N + i] }
            cands.append(Cand(cls: bestC, score: best, box: box, coeffs: coeffs))
        }

        // non-maximum suppression
        cands.sort { $0.score > $1.score }
        var kept: [Cand] = []
        for c in cands {
            if kept.count >= maxDetections { break }
            if kept.allSatisfy({ iouOf($0.box, c.box) < iou }) { kept.append(c) }
        }

        let area = mh * mw
        return kept.map { c in
            var mask = [Bool](repeating: false, count: area)
            let x0 = max(0, Int(Float(c.box.minX) * Float(mw))), x1 = min(mw, Int(ceil(Float(c.box.maxX) * Float(mw))))
            let y0 = max(0, Int(Float(c.box.minY) * Float(mh))), y1 = min(mh, Int(ceil(Float(c.box.maxY) * Float(mh))))
            if x0 < x1 && y0 < y1 {
                for y in y0..<y1 {
                    for x in x0..<x1 {
                        let j = y * mw + x
                        var v: Float = 0
                        for k in 0..<nm { v += c.coeffs[k] * P[k * area + j] }
                        mask[j] = v > 0  // sigmoid(v) > 0.5
                    }
                }
            }
            return Detection(classId: c.cls, score: c.score, box: c.box, mask: mask, maskWidth: mw, maskHeight: mh)
        }
    }

    private static func iouOf(_ a: CGRect, _ b: CGRect) -> Float {
        let i = a.intersection(b)
        if i.isNull { return 0 }
        let inter = i.width * i.height
        return Float(inter / (a.width * a.height + b.width * b.height - inter))
    }

    /// Row-major Float copy that honours the array's strides and Float16/Float32/Double storage.
    static func contiguous(_ a: MLMultiArray) -> [Float] {
        let shape = a.shape.map { $0.intValue }
        let strides = a.strides.map { $0.intValue }
        let count = shape.reduce(1, *)
        var out = [Float](repeating: 0, count: count)
        let read: (Int) -> Float
        switch a.dataType {
        case .float16:
            let p = a.dataPointer.assumingMemoryBound(to: Float16.self)
            read = { Float(p[$0]) }
        case .double:
            let p = a.dataPointer.assumingMemoryBound(to: Double.self)
            read = { Float(p[$0]) }
        default:
            let p = a.dataPointer.assumingMemoryBound(to: Float.self)
            read = { p[$0] }
        }
        var idx = [Int](repeating: 0, count: shape.count)
        for n in 0..<count {
            var off = 0
            for d in 0..<shape.count { off += idx[d] * strides[d] }
            out[n] = read(off)
            var d = shape.count - 1
            while d >= 0 {
                idx[d] += 1
                if idx[d] < shape[d] { break }
                idx[d] = 0
                d -= 1
            }
        }
        return out
    }
}
