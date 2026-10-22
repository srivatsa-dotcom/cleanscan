package com.cleanscan.app

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * Decodes raw Ultralytics YOLO-seg outputs (exported with nms=False). Pure Kotlin so it
 * can be unit-tested on the JVM.
 *
 *  det:   [C, N] with C = 4 + numClasses + numMasks (cx, cy, w, h in input pixels)
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
            val cx = det[i] / inputW
            val cy = det[anchors + i] / inputH
            val w = det[2 * anchors + i] / inputW
            val h = det[3 * anchors + i] / inputH
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
