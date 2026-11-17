package com.cleanscan.app

import kotlin.math.PI
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.min

/*
 * Object selection and guided capture for RL scenes (docs/RL_SCENES_PLAN.md), the parts
 * without Android dependencies so they run and are tested on the plain JVM
 * (tools/app-logic-test). The iOS app has the same logic in ObjectCapture.swift.
 */

/** COCO class names in model order (YOLO-seg models trained on COCO). */
val COCO_NAMES = arrayOf(
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
)

/** How an object becomes interactive in the simulation. */
enum class CaptureMode(val key: String, val stages: List<String>, val label: String) {
    PICKABLE("pickable", listOf("scan"), "Pick up"),
    ARTICULATED("articulated", listOf("closed", "open"), "Opens"),
    TRANSPARENT("transparent", listOf("profile", "top"), "Glass");

    companion object {
        private val ARTICULATED_CLASSES = setOf("refrigerator", "fridge", "cupboard", "cabinet", "drawer",
            "oven", "microwave", "door", "dishwasher", "toilet")
        private val TRANSPARENT_CLASSES = setOf("wine glass", "glass", "tumbler", "steel tumbler")

        /** Same table as the desktop pipeline (pipeline/cleanscan/objects.py). */
        fun forClass(cls: String): CaptureMode {
            val c = cls.trim().lowercase()
            return when (c) {
                in ARTICULATED_CLASSES -> ARTICULATED
                in TRANSPARENT_CLASSES -> TRANSPARENT
                else -> PICKABLE
            }
        }

        fun of(key: String) = values().first { it.key == key }
    }
}

/** Classes the live detector reports as candidate objects (not people/animals, not furniture
 * that can only be part of the static room). */
val OBJECT_CLASSES: IntArray = (0 until COCO_NAMES.size).filter { c ->
    c !in YoloSegDecoder.LIVING && COCO_NAMES[c] !in setOf("bench", "chair", "couch", "bed", "dining table",
        "tv", "potted plant", "car", "bus", "train", "truck", "airplane", "boat", "bicycle", "motorcycle",
        "traffic light", "fire hydrant", "stop sign", "parking meter")
}.toIntArray()

/** A box in the portrait-normalised image (top-left origin), as the detector reports it. */
data class Box(val x0: Float, val y0: Float, val x1: Float, val y1: Float) {
    val area get() = max(0f, x1 - x0) * max(0f, y1 - y0)
    val cx get() = (x0 + x1) / 2
    val cy get() = (y0 + y1) / 2

    fun iou(o: Box): Float {
        val iw = min(x1, o.x1) - max(x0, o.x0)
        val ih = min(y1, o.y1) - max(y0, o.y0)
        if (iw <= 0 || ih <= 0) return 0f
        val inter = iw * ih
        return inter / (area + o.area - inter)
    }
}

class Track(val id: Int, val classId: Int, val firstFrame: Int) {
    var frames = 0
    var lastBox: Box = Box(0f, 0f, 0f, 0f)
    var lastSeen = 0.0
    var bestScore = 0f
    var bestFrame = -1
    val className get() = COCO_NAMES.getOrElse(classId) { "object" }
}

/**
 * Keeps object identities across frames: a detection continues the track of the same class
 * whose last box overlaps it most (IoU), if that track was seen recently; otherwise it
 * starts a new track. Phone motion between recorded frames (5 fps) is small enough for this.
 */
class ObjectTracker(private val iouThreshold: Float = 0.25f, private val maxGapSeconds: Double = 1.5,
                    firstId: Int = 1) {
    private var nextId = firstId
    val tracks = LinkedHashMap<Int, Track>()

    /** Returns the track id for every detection (same order). */
    fun update(frameIndex: Int, time: Double, classes: IntArray, scores: FloatArray, boxes: List<Box>): IntArray {
        val ids = IntArray(boxes.size) { -1 }
        val taken = HashSet<Int>()
        val order = boxes.indices.sortedByDescending { scores[it] }
        for (i in order) {
            var best: Track? = null
            var bestIou = iouThreshold
            for (t in tracks.values) {
                if (t.classId != classes[i] || t.id in taken || time - t.lastSeen > maxGapSeconds) continue
                val iou = t.lastBox.iou(boxes[i])
                if (iou > bestIou) { bestIou = iou; best = t }
            }
            val t = best ?: Track(nextId++, classes[i], frameIndex).also { tracks[it.id] = it }
            taken.add(t.id)
            t.frames++
            t.lastBox = boxes[i]
            t.lastSeen = time
            // the best view for the thumbnail: confident and not cut off at the image border
            val inside = boxes[i].x0 > 0.02f && boxes[i].y0 > 0.02f && boxes[i].x1 < 0.98f && boxes[i].y1 < 0.98f
            val s = scores[i] * (if (inside) 1f else 0.7f)
            if (s > t.bestScore) { t.bestScore = s; t.bestFrame = frameIndex }
            ids[i] = t.id
        }
        return ids
    }

    /** Tracks seen in enough frames to be offered to the user. */
    fun confirmed(minFrames: Int = 3): List<Track> = tracks.values.filter { it.frames >= minFrames }
}

/** Fraction of viewing directions (azimuth bins) covered around an object. */
class CoverageRing(val bins: Int = 36) {
    private val hit = BooleanArray(bins)

    /** Camera and object positions in the AR world (Y up). */
    fun add(camX: Float, camZ: Float, objX: Float, objZ: Float) {
        val a = atan2((camZ - objZ).toDouble(), (camX - objX).toDouble())
        val b = (((a + PI) / (2 * PI)) * bins).toInt().coerceIn(0, bins - 1)
        hit[b] = true
    }

    val fraction get() = hit.count { it }.toFloat() / bins
    fun covered(bin: Int) = hit[bin]
}

/** Camera pitch in degrees (0 = level, -90 = looking straight down) from a row-major
 * camera-to-world matrix (OpenGL camera: it looks along -Z). */
fun cameraPitchDeg(rowMajor: FloatArray): Float {
    val fy = -rowMajor[6]  // world y of the camera's -Z axis
    return Math.toDegrees(asin(fy.coerceIn(-1f, 1f).toDouble())).toFloat()
}

/** What the user chose on the selection screen. */
data class SelectedObject(
    val id: Int, val cls: String, val name: String, var mode: CaptureMode, val source: String,
    val seedFrame: Int? = null, val seedU: Float? = null, val seedV: Float? = null,
) {
    val captures = LinkedHashMap<String, String>()
}

/** JSON written next to the scan (docs/capture_format.md); hand-written to stay JVM-only. */
object ObjectJson {
    fun str(s: String) = "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\""

    fun tracks(tracks: List<Track>): String = tracks.joinToString(",\n ", "[\n ", "\n]\n") { t ->
        "{\"id\": ${t.id}, \"class\": ${str(t.className)}, \"coco_id\": ${t.classId}, " +
            "\"score\": ${"%.3f".format(java.util.Locale.US, t.bestScore)}, \"frames\": ${t.frames}, " +
            "\"first_frame\": ${t.firstFrame}, \"best_frame\": ${t.bestFrame}, \"thumb\": ${str("thumbs/${t.id}.jpg")}}"
    }

    fun selection(objects: List<SelectedObject>): String = objects.joinToString(",\n  ", "{\"objects\": [\n  ", "\n]}\n") { o ->
        val caps = o.captures.entries.joinToString(", ", "{", "}") { "${str(it.key)}: ${str(it.value)}" }
        val seed = if (o.seedFrame != null) ", \"seed\": {\"frame\": ${o.seedFrame}, " +
            "\"u\": ${"%.4f".format(java.util.Locale.US, o.seedU)}, \"v\": ${"%.4f".format(java.util.Locale.US, o.seedV)}}" else ""
        "{\"id\": ${o.id}, \"class\": ${str(o.cls)}, \"name\": ${str(o.name)}, \"mode\": ${str(o.mode.key)}, " +
            "\"source\": ${str(o.source)}, \"captures\": $caps$seed}"
    }
}

/** Portrait-normalised detection masks -> 16-bit instance labels over the landscape depth image. */
fun instanceLabels(dets: List<YoloSegDecoder.Detection>, ids: IntArray, width: Int, height: Int): ShortArray {
    val out = ShortArray(width * height)
    if (dets.isEmpty()) return out
    for (y in 0 until height) {
        val xp = 1f - (y + 0.5f) / height
        for (x in 0 until width) {
            val yp = (x + 0.5f) / width
            for (k in dets.indices) {
                if (ids[k] > 0 && dets[k].contains(xp, yp)) { out[y * width + x] = ids[k].toShort(); break }
            }
        }
    }
    return out
}

/** Portrait touch point (0..1, top-left origin) -> sensor-image (landscape) coordinates. */
fun portraitToSensor(xp: Float, yp: Float): Pair<Float, Float> = Pair(yp, 1f - xp)
