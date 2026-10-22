package com.cleanscan.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

/**
 * People + animal segmentation with a YOLO-seg TFLite model placed in
 * app/src/main/assets/living_seg.tflite (see tools/export_models.py).
 */
class LivingSegmenter(context: Context) {
    private val interpreter: Interpreter? = try {
        val fd = context.assets.openFd(MODEL)
        val buf = FileInputStream(fd.fileDescriptor).channel
            .map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
        Interpreter(buf, Interpreter.Options().setNumThreads(4))
    } catch (e: Exception) {
        Log.w(TAG, "no segmentation model ($MODEL): ${e.message}")
        null
    }

    val available get() = interpreter != null

    /**
     * @param sensorImage camera image in sensor (landscape) orientation
     * @return mask (0 / 255) at depth resolution, depth image is also sensor-oriented
     */
    fun segment(sensorImage: Bitmap, depthW: Int, depthH: Int): ByteArray {
        val out = ByteArray(depthW * depthH)
        val tfl = interpreter ?: return out
        val inShape = tfl.getInputTensor(0).shape()  // [1, H, W, 3]
        val inH = inShape[1]
        val inW = inShape[2]

        // Phone is held in portrait: rotate so people are upright for the detector.
        val upright = Bitmap.createBitmap(sensorImage, 0, 0, sensorImage.width, sensorImage.height,
            Matrix().apply { postRotate(90f) }, true)
        val scaled = Bitmap.createScaledBitmap(upright, inW, inH, true)
        val input = ByteBuffer.allocateDirect(4 * inW * inH * 3).order(ByteOrder.nativeOrder())
        val px = IntArray(inW * inH)
        scaled.getPixels(px, 0, inW, 0, 0, inW, inH)
        for (p in px) {
            input.putFloat(((p shr 16) and 0xFF) / 255f)
            input.putFloat(((p shr 8) and 0xFF) / 255f)
            input.putFloat((p and 0xFF) / 255f)
        }
        input.rewind()

        // Identify outputs by rank: detections are 3-D, prototypes 4-D.
        var detIdx = 0
        var protoIdx = 1
        for (i in 0 until tfl.outputTensorCount) {
            if (tfl.getOutputTensor(i).shape().size == 4) protoIdx = i else detIdx = i
        }
        val dShape = tfl.getOutputTensor(detIdx).shape()
        val pShape = tfl.getOutputTensor(protoIdx).shape()
        val detBuf = ByteBuffer.allocateDirect(4 * dShape.fold(1) { a, b -> a * b }).order(ByteOrder.nativeOrder())
        val protoBuf = ByteBuffer.allocateDirect(4 * pShape.fold(1) { a, b -> a * b }).order(ByteOrder.nativeOrder())
        tfl.runForMultipleInputsOutputs(arrayOf(input), mapOf(detIdx to detBuf, protoIdx to protoBuf))
        detBuf.rewind()
        protoBuf.rewind()

        // TFLite prototypes are NHWC [1, mh, mw, nm]; transpose to [nm, mh, mw].
        val channelsLast = pShape[3] < pShape[1]
        val nm = if (channelsLast) pShape[3] else pShape[1]
        val mh = if (channelsLast) pShape[1] else pShape[2]
        val mw = if (channelsLast) pShape[2] else pShape[3]
        val rawProto = FloatArray(nm * mh * mw).also { protoBuf.asFloatBuffer().get(it) }
        val proto = if (!channelsLast) rawProto else FloatArray(rawProto.size).also { t ->
            for (y in 0 until mh) for (x in 0 until mw) for (k in 0 until nm)
                t[k * mh * mw + y * mw + x] = rawProto[(y * mw + x) * nm + k]
        }

        // detections may be [1, C, N] or [1, N, C]
        var c = dShape[1]
        var n = dShape[2]
        var det = FloatArray(c * n).also { detBuf.asFloatBuffer().get(it) }
        if (c > n) {
            val t = FloatArray(det.size)
            for (i in 0 until c) for (j in 0 until n) t[j * c + i] = det[i * n + j]
            det = t
            c = dShape[2]; n = dShape[1]
        }
        val dets = YoloSegDecoder.decode(det, c, n, proto, nm, mh, mw, inW, inH)
        YoloSegDecoder.rasterize(dets, depthW, depthH, out)
        return out
    }

    companion object {
        private const val TAG = "LivingSegmenter"
        const val MODEL = "living_seg.tflite"
    }
}
