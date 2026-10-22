package com.cleanscan.app

import android.content.Context
import android.os.Build
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.io.FileWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

/** One frame's data, copied out of ARCore so the ARCore images can be closed at once. */
class CapturedFrame(
    val timestamp: Double,
    val jpeg: ByteArray, val colorW: Int, val colorH: Int,
    val colorIntrinsics: FloatArray,  // fx, fy, cx, cy for the CPU image
    val depthMM: ShortArray, val confidence: ByteArray?, val depthW: Int, val depthH: Int,
    val depthIntrinsics: FloatArray,  // fx, fy, cx, cy for the depth image
    val cameraToWorld: FloatArray,  // row-major 4x4
)

/** Writes a session in the CleanScan capture format (docs/capture_format.md). */
class SessionWriter private constructor(val dir: File) {
    private val frames = FileWriter(File(dir, "frames.jsonl"), true)
    private var index = 0

    fun write(f: CapturedFrame, mask: ByteArray) {
        val name = "%06d".format(index)
        File(dir, "color/$name.jpg").writeBytes(f.jpeg)
        PngWriter.writeGray16(File(dir, "depth/$name.png"), f.depthW, f.depthH, f.depthMM)
        PngWriter.writeGray8(File(dir, "mask/$name.png"), f.depthW, f.depthH, mask)
        val rec = JSONObject()
            .put("index", index)
            .put("timestamp", f.timestamp)
            .put("color", "color/$name.jpg")
            .put("depth", "depth/$name.png")
            .put("mask", "mask/$name.png")
            .put("color_intrinsics", JSONArray(f.colorIntrinsics.map { it.toDouble() }))
            .put("color_size", JSONArray(listOf(f.colorW, f.colorH)))
            .put("depth_intrinsics", JSONArray(f.depthIntrinsics.map { it.toDouble() }))
            .put("depth_size", JSONArray(listOf(f.depthW, f.depthH)))
            .put("camera_to_world", JSONArray(f.cameraToWorld.map { it.toDouble() }))
        f.confidence?.let {
            PngWriter.writeGray8(File(dir, "confidence/$name.png"), f.depthW, f.depthH, it)
            rec.put("confidence", "confidence/$name.png")
        }
        frames.write(rec.toString() + "\n")
        index++
    }

    fun finish() = frames.close()

    companion object {
        fun scansDir(context: Context) = File(context.getExternalFilesDir(null), "Scans")

        fun create(context: Context): SessionWriter {
            val stamp = SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(Date())
            val dir = File(scansDir(context), "Scan-$stamp")
            for (sub in listOf("color", "depth", "confidence", "mask")) File(dir, sub).mkdirs()
            File(dir, "session.json").writeText(
                JSONObject()
                    .put("format", "cleanscan-capture")
                    .put("version", 1)
                    .put("platform", "android")
                    .put("device", "${Build.MANUFACTURER} ${Build.MODEL}")
                    .put("depth_unit", 0.001)
                    .put("confidence_levels", 256)
                    // ARCore raw-depth confidence is conservative; 0.3 keeps most good points
                    .put("recommended_min_confidence", 0.3)
                    .put("world_up", "y")
                    .put("camera_convention", "opengl")
                    .toString(2)
            )
            return SessionWriter(dir)
        }

        fun list(context: Context): List<File> =
            scansDir(context).listFiles { f -> f.isDirectory }?.sortedByDescending { it.name } ?: emptyList()

        fun frameCount(dir: File): Int =
            File(dir, "frames.jsonl").takeIf { it.exists() }?.useLines { it.count() } ?: 0

        /** Zip a session into the cache dir so it can be shared (FileProvider). */
        fun zip(context: Context, dir: File): File {
            val outDir = File(context.cacheDir, "exports").apply { mkdirs() }
            val zip = File(outDir, "${dir.name}.zip")
            ZipOutputStream(FileOutputStream(zip).buffered()).use { z ->
                dir.walkTopDown().filter { it.isFile }.forEach { f ->
                    z.putNextEntry(ZipEntry("${dir.name}/${f.relativeTo(dir).path}"))
                    f.inputStream().use { it.copyTo(z) }
                    z.closeEntry()
                }
            }
            return zip
        }
    }
}
