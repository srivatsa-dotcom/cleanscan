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
        val mask = ByteArray(depthW * depthH)
        val tfl = interpreter ?: return mask
        // Input is [1, 3, H, W] (LiteRT export) or [1, H, W, 3] (older TF exports).
        val layout = YoloSegDecoder.InputLayout.of(tfl.getInputTensor(0).shape())

        // Phone is held in portrait: rotate so people are upright for the detector.
        val upright = Bitmap.createBitmap(sensorImage, 0, 0, sensorImage.width, sensorImage.height,
            Matrix().apply { postRotate(90f) }, true)
        val scaled = Bitmap.createScaledBitmap(upright, layout.width, layout.height, true)
        val px = IntArray(layout.width * layout.height)
        scaled.getPixels(px, 0, layout.width, 0, 0, layout.width, layout.height)
        val packed = YoloSegDecoder.packInput(px, layout)
        val input = ByteBuffer.allocateDirect(4 * packed.size).order(ByteOrder.nativeOrder())
        input.asFloatBuffer().put(packed)

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
        val rawDet = FloatArray(detBuf.capacity() / 4).also { detBuf.asFloatBuffer().get(it) }
        val rawProto = FloatArray(protoBuf.capacity() / 4).also { protoBuf.asFloatBuffer().get(it) }

        val out = YoloSegDecoder.Outputs.of(dShape, rawDet, pShape, rawProto)
        val dets = YoloSegDecoder.decode(out.det, out.channels, out.anchors, out.proto, out.numMasks,
            out.maskH, out.maskW, layout.width, layout.height)
        YoloSegDecoder.rasterize(dets, depthW, depthH, mask)
        return mask
    }

    companion object {
        private const val TAG = "LivingSegmenter"
        const val MODEL = "living_seg.tflite"
    }
}
