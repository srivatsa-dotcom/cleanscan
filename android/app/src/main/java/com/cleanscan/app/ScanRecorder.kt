package com.cleanscan.app

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import com.google.ar.core.Frame
import com.google.ar.core.TrackingState
import com.google.ar.core.exceptions.NotYetAvailableException
import java.io.ByteArrayOutputStream
import java.nio.ByteOrder
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.acos
import kotlin.math.sqrt

/**
 * Called on the GL thread for every ARCore frame while recording. Copies colour, raw
 * depth, confidence and pose out of ARCore, then segments + writes on a worker thread.
 */
class ScanRecorder(
    private val writer: SessionWriter,
    private val segmenter: LivingSegmenter,
    private val onFrameWritten: (count: Int, maskedFraction: Float, mask: Bitmap?) -> Unit,
    private val onError: (String) -> Unit,
) {
    private val worker = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var lastPose: FloatArray? = null
    private var lastTimeNs = 0L
    private var count = 0

    fun onFrame(frame: Frame) {
        val camera = frame.camera
        if (camera.trackingState != TrackingState.TRACKING || busy.get()) return
        val t = frame.timestamp
        if (t - lastTimeNs < MIN_INTERVAL_NS) return

        val pose = FloatArray(16).also { camera.pose.toMatrix(it, 0) }  // column-major
        lastPose?.let { last ->
            if (t - lastTimeNs < MAX_INTERVAL_NS && !movedEnough(last, pose)) return
        }

        val captured = try {
            copyFrame(frame, pose) ?: return
        } catch (e: NotYetAvailableException) {
            return  // depth not ready for this frame yet
        }
        lastPose = pose
        lastTimeNs = t
        busy.set(true)
        worker.execute {
            try {
                val bmp = BitmapFactory.decodeByteArray(captured.jpeg, 0, captured.jpeg.size)
                val mask = segmenter.segment(bmp, captured.depthW, captured.depthH)
                writer.write(captured, mask)
                count++
                val masked = mask.count { it != 0.toByte() }.toFloat() / mask.size
                onFrameWritten(count, masked, maskPreview(mask, captured.depthW, captured.depthH))
            } catch (e: Exception) {
                onError("write failed: ${e.message}")
            } finally {
                busy.set(false)
            }
        }
    }

    fun finish(done: () -> Unit) {
        worker.execute {
            writer.finish()
            done()
        }
        worker.shutdown()
    }

    private fun copyFrame(frame: Frame, poseColMajor: FloatArray): CapturedFrame? {
        val camera = frame.camera
        frame.acquireRawDepthImage16Bits().use { depth ->
            frame.acquireRawDepthConfidenceImage().use { conf ->
                frame.acquireCameraImage().use { color ->
                    val dw = depth.width
                    val dh = depth.height
                    val depthMM = ShortArray(dw * dh)
                    val dPlane = depth.planes[0]
                    val dBuf = dPlane.buffer.order(ByteOrder.nativeOrder())
                    for (y in 0 until dh) for (x in 0 until dw) {
                        depthMM[y * dw + x] = dBuf.getShort(y * dPlane.rowStride + x * dPlane.pixelStride)
                    }
                    val confidence = ByteArray(dw * dh)
                    val cPlane = conf.planes[0]
                    for (y in 0 until dh) for (x in 0 until dw) {
                        confidence[y * dw + x] = cPlane.buffer.get(y * cPlane.rowStride + x * cPlane.pixelStride)
                    }

                    val jpeg = yuvToJpeg(color)
                    val ci = camera.imageIntrinsics  // for the CPU image
                    val cf = ci.focalLength
                    val cp = ci.principalPoint
                    // Raw depth is aligned with the camera texture; scale its intrinsics.
                    val ti = camera.textureIntrinsics
                    val tf = ti.focalLength
                    val tp = ti.principalPoint
                    val tdim = ti.imageDimensions
                    val sx = dw.toFloat() / tdim[0]
                    val sy = dh.toFloat() / tdim[1]

                    val rowMajor = FloatArray(16)
                    for (r in 0 until 4) for (c in 0 until 4) rowMajor[r * 4 + c] = poseColMajor[c * 4 + r]
                    return CapturedFrame(
                        timestamp = frame.timestamp / 1e9,
                        jpeg = jpeg, colorW = color.width, colorH = color.height,
                        colorIntrinsics = floatArrayOf(cf[0], cf[1], cp[0], cp[1]),
                        depthMM = depthMM, confidence = confidence, depthW = dw, depthH = dh,
                        depthIntrinsics = floatArrayOf(tf[0] * sx, tf[1] * sy, tp[0] * sx, tp[1] * sy),
                        cameraToWorld = rowMajor,
                    )
                }
            }
        }
    }

    private fun movedEnough(a: FloatArray, b: FloatArray): Boolean {
        val dx = a[12] - b[12]
        val dy = a[13] - b[13]
        val dz = a[14] - b[14]
        if (sqrt(dx * dx + dy * dy + dz * dz) > MIN_TRANSLATION) return true
        // angle between rotations: trace(Ra^T Rb) = 1 + 2 cos(theta)
        var trace = 0f
        for (i in 0 until 3) for (k in 0 until 3) trace += a[i * 4 + k] * b[i * 4 + k]
        val angle = Math.toDegrees(acos(((trace - 1) / 2).coerceIn(-1f, 1f)).toDouble())
        return angle > MIN_ROTATION_DEG
    }

    private fun maskPreview(mask: ByteArray, w: Int, h: Int): Bitmap {
        val px = IntArray(w * h) { if (mask[it] != 0.toByte()) 0xA0FF0000.toInt() else 0x00000000 }
        val bmp = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        val m = android.graphics.Matrix().apply { postRotate(90f) }
        return Bitmap.createBitmap(bmp, 0, 0, w, h, m, false)
    }

    companion object {
        private const val MIN_INTERVAL_NS = 200_000_000L  // <= 5 fps
        private const val MAX_INTERVAL_NS = 500_000_000L  // keep recording while standing still
        private const val MIN_TRANSLATION = 0.01f
        private const val MIN_ROTATION_DEG = 2.0

        /** YUV_420_888 -> NV21 -> JPEG, honouring row and pixel strides. */
        fun yuvToJpeg(image: Image, quality: Int = 90): ByteArray {
            val w = image.width
            val h = image.height
            val nv21 = ByteArray(w * h * 3 / 2)
            val y = image.planes[0]
            var o = 0
            for (r in 0 until h) for (c in 0 until w) nv21[o++] = y.buffer.get(r * y.rowStride + c * y.pixelStride)
            val u = image.planes[1]
            val v = image.planes[2]
            for (r in 0 until h / 2) for (c in 0 until w / 2) {
                nv21[o++] = v.buffer.get(r * v.rowStride + c * v.pixelStride)
                nv21[o++] = u.buffer.get(r * u.rowStride + c * u.pixelStride)
            }
            val out = ByteArrayOutputStream()
            YuvImage(nv21, ImageFormat.NV21, w, h, null).compressToJpeg(Rect(0, 0, w, h), quality, out)
            return out.toByteArray()
        }
    }
}
