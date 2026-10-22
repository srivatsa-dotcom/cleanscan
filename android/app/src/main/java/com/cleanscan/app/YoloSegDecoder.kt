package com.cleanscan.app

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Decodes raw Ultralytics YOLO-seg outputs (exported with nms=False). Pure Kotlin so it
 * can be unit-tested on the JVM.
 *
 *  det:   [C, N] with C = 4 + numClasses + numMasks (cx, cy, w, h in input pixels or 0..1)
 *  proto: [numMasks, mh, mw]
 */
object YoloSegDecoder {
    /** COCO: person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe */
    val LIVING = intArrayOf(0, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)

    class Detection(
        val classId: Int, val score: Float,
        val x0: Float, val y0: Float, val x1: Float, val y1: Float,  // normalised, top-left origin
        val mask: BooleanArray, val mw: Int, val mh: Int,
    ) {
        fun contains(x: Float, y: Float): Boolean {
            if (x < x0 || x > x1 || y < y0 || y > y1) return false
            val mx = min(mw - 1, max(0, (x * mw).toInt()))
            val my = min(mh - 1, max(0, (y * mh).toInt()))
            return mask[my * mw + mx]
        }
    }

    /** Model input tensor layout. */
    class InputLayout(val width: Int, val height: Int, val channelsFirst: Boolean) {
        companion object {
            fun of(shape: IntArray): InputLayout =
                if (shape[1] == 3) InputLayout(shape[3], shape[2], true)  // [1, 3, H, W]
                else InputLayout(shape[2], shape[1], false)  // [1, H, W, 3]
        }
    }

    /** ARGB pixels (row-major, width x height of the layout) -> RGB floats in 0..1. */
    fun packInput(px: IntArray, layout: InputLayout): FloatArray {
        val n = layout.width * layout.height
        val out = FloatArray(3 * n)
        for (i in 0 until n) {
            val p = px[i]
            val r = ((p shr 16) and 0xFF) / 255f
            val g = ((p shr 8) and 0xFF) / 255f
            val b = (p and 0xFF) / 255f
            if (layout.channelsFirst) {
                out[i] = r; out[n + i] = g; out[2 * n + i] = b
            } else {
                out[3 * i] = r; out[3 * i + 1] = g; out[3 * i + 2] = b
            }
        }
        return out
    }

    /** Raw output tensors normalised to det [C, N] and proto [numMasks, maskH, maskW]. */
    class Outputs(val det: FloatArray, val channels: Int, val anchors: Int,
                  val proto: FloatArray, val numMasks: Int, val maskH: Int, val maskW: Int) {
        companion object {
            fun of(detShape: IntArray, det: FloatArray, protoShape: IntArray, proto: FloatArray): Outputs {
                // prototypes: [1, nm, mh, mw] or [1, mh, mw, nm]
                val channelsLast = protoShape[3] < protoShape[1]
                val nm = if (channelsLast) protoShape[3] else protoShape[1]
                val mh = if (channelsLast) protoShape[1] else protoShape[2]
                val mw = if (channelsLast) protoShape[2] else protoShape[3]
                val p = if (!channelsLast) proto else FloatArray(proto.size).also { t ->
                    for (y in 0 until mh) for (x in 0 until mw) for (k in 0 until nm)
                        t[k * mh * mw + y * mw + x] = proto[(y * mw + x) * nm + k]
                }
                // detections: [1, C, N] or [1, N, C]
                var c = detShape[1]
                var n = detShape[2]
                var d = det
                if (c > n) {
                    d = FloatArray(det.size)
                    for (i in 0 until c) for (j in 0 until n) d[j * c + i] = det[i * n + j]
                    c = detShape[2]; n = detShape[1]
                }
                return Outputs(d, c, n, p, nm, mh, mw)
            }
        }
    }

    fun decode(
        det: FloatArray, channels: Int, anchors: Int,
        proto: FloatArray, numMasks: Int, mh: Int, mw: Int,
        inputW: Int, inputH: Int,
        classes: IntArray = LIVING, confidence: Float = 0.3f, iou: Float = 0.5f, maxDet: Int = 30,
    ): List<Detection> {
        val nc = channels - 4 - numMasks
        if (nc <= 0) return emptyList()
        data class Cand(val cls: Int, val score: Float, val x0: Float, val y0: Float,
                        val x1: Float, val y1: Float, val coeffs: FloatArray)

        // TFLite/LiteRT exports emit boxes normalised to 0..1, Core ML exports in input pixels.
        // Anchor centres span the whole image, so the largest cx tells the two apart.
        var maxCx = 0f
        for (i in 0 until anchors) maxCx = max(maxCx, det[i])
        val sx = if (maxCx <= 2f) 1f else inputW.toFloat()
        val sy = if (maxCx <= 2f) 1f else inputH.toFloat()

        val cands = ArrayList<Cand>()
        for (i in 0 until anchors) {
            var best = 0f
            var bestC = -1
            for (c in classes) {
                if (c >= nc) continue
                val s = det[(4 + c) * anchors + i]
                if (s > best) { best = s; bestC = c }
            }
            if (bestC < 0 || best < confidence) continue
            val cx = det[i] / sx
            val cy = det[anchors + i] / sy
            val w = det[2 * anchors + i] / sx
            val h = det[3 * anchors + i] / sy
            val coeffs = FloatArray(numMasks) { det[(4 + nc + it) * anchors + i] }
            cands.add(Cand(bestC, best, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, coeffs))
        }
        cands.sortByDescending { it.score }
        val kept = ArrayList<Cand>()
        for (c in cands) {
            if (kept.size >= maxDet) break
            if (kept.all { iou(it.x0, it.y0, it.x1, it.y1, c.x0, c.y0, c.x1, c.y1) < iou }) kept.add(c)
        }

        val area = mh * mw
        return kept.map { c ->
            val mask = BooleanArray(area)
            val bx0 = max(0, (c.x0 * mw).toInt()); val bx1 = min(mw, ceil(c.x1 * mw).toInt())
            val by0 = max(0, (c.y0 * mh).toInt()); val by1 = min(mh, ceil(c.y1 * mh).toInt())
            for (y in by0 until by1) for (x in bx0 until bx1) {
                val j = y * mw + x
                var v = 0f
                for (k in 0 until numMasks) v += c.coeffs[k] * proto[k * area + j]
                mask[j] = v > 0f  // sigmoid(v) > 0.5
            }
            Detection(c.cls, c.score, c.x0, c.y0, c.x1, c.y1, mask, mw, mh)
        }
    }

    private fun iou(ax0: Float, ay0: Float, ax1: Float, ay1: Float,
                    bx0: Float, by0: Float, bx1: Float, by1: Float): Float {
        val iw = min(ax1, bx1) - max(ax0, bx0)
        val ih = min(ay1, by1) - max(ay0, by0)
        if (iw <= 0 || ih <= 0) return 0f
        val inter = iw * ih
        return inter / ((ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)
    }

    /**
     * Rasterise detections (made on the portrait-rotated image) into a mask over the
     * landscape sensor-oriented depth image. Portrait (xp, yp) <-> landscape (xl, yl):
     * xl = yp, yl = 1 - xp.
     */
    fun rasterize(dets: List<Detection>, width: Int, height: Int, out: ByteArray) {
        if (dets.isEmpty()) return
        for (y in 0 until height) {
            val xp = 1f - (y + 0.5f) / height
            for (x in 0 until width) {
                val yp = (x + 0.5f) / width
                if (dets.any { it.contains(xp, yp) }) out[y * width + x] = 0xFF.toByte()
            }
        }
    }
}
