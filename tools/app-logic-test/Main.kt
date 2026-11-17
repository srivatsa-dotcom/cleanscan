// JVM checks for the app's object-capture logic (android/.../ObjectCapture.kt), plus a fixture
// scan whose tracks.json / selection.json the Python pipeline must read. Run: ./run.sh <out_dir>
import com.cleanscan.app.Box
import com.cleanscan.app.COCO_NAMES
import com.cleanscan.app.CaptureMode
import com.cleanscan.app.CoverageRing
import com.cleanscan.app.ObjectJson
import com.cleanscan.app.ObjectTracker
import com.cleanscan.app.SelectedObject
import com.cleanscan.app.YoloSegDecoder
import com.cleanscan.app.cameraPitchDeg
import com.cleanscan.app.instanceLabels
import com.cleanscan.app.portraitToSensor
import java.io.File
import kotlin.math.cos
import kotlin.math.sin

fun check(ok: Boolean, msg: String) { if (!ok) throw AssertionError(msg); println("ok  $msg") }

fun main(args: Array<String>) {
    check(COCO_NAMES.size == 80 && COCO_NAMES[39] == "bottle" && COCO_NAMES[72] == "refrigerator", "COCO names")
    check(CaptureMode.forClass("Refrigerator") == CaptureMode.ARTICULATED, "fridge opens")
    check(CaptureMode.forClass("wine glass") == CaptureMode.TRANSPARENT, "wine glass is transparent")
    check(CaptureMode.forClass("pressure cooker") == CaptureMode.PICKABLE, "unknown classes are pickable")

    // tracker: a bottle drifting right keeps its id, a second bottle gets a new one, a gap ends a track
    val tr = ObjectTracker()
    var t = 0.0
    var ids = IntArray(0)
    for (i in 0 until 5) {
        val dx = 0.02f * i
        ids = tr.update(i, t, intArrayOf(39, 39), floatArrayOf(0.8f, 0.6f),
            listOf(Box(0.2f + dx, 0.4f, 0.3f + dx, 0.6f), Box(0.6f, 0.4f, 0.7f, 0.6f)))
        t += 0.2
    }
    check(ids.toList() == listOf(1, 2), "two bottles keep their ids while the phone moves")
    val later = tr.update(10, t + 3.0, intArrayOf(39), floatArrayOf(0.9f), listOf(Box(0.2f, 0.4f, 0.3f, 0.6f)))
    check(later[0] == 3, "an object seen again after a long gap is a new track")
    check(tr.confirmed().map { it.id } == listOf(1, 2), "only tracks seen in 3+ frames are offered")

    // coverage ring: walking a full circle around an object covers it, a quarter does not
    val ring = CoverageRing()
    for (k in 0 until 90) { val a = 2 * Math.PI * k / 360; ring.add(cos(a).toFloat(), sin(a).toFloat(), 0f, 0f) }
    check(ring.fraction in 0.24f..0.28f, "a quarter turn covers ~25 % (${ring.fraction})")
    for (k in 0 until 360) { val a = 2 * Math.PI * k / 360; ring.add(cos(a).toFloat(), sin(a).toFloat(), 0f, 0f) }
    check(ring.fraction == 1f, "a full turn covers 100 %")

    // pitch: identity pose looks along -Z (level); rotated -90 deg about X looks down
    val level = floatArrayOf(1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f, 1f)
    val down = floatArrayOf(1f, 0f, 0f, 0f, 0f, 0f, 1f, 0f, 0f, -1f, 0f, 0f, 0f, 0f, 0f, 1f)
    check(kotlin.math.abs(cameraPitchDeg(level)) < 1e-3, "level phone has pitch 0")
    check(kotlin.math.abs(cameraPitchDeg(down) + 90f) < 1e-3, "phone flat above a table has pitch -90")

    // instance labels: a detection covering the left half of the portrait image lands in the
    // bottom half of the landscape sensor image (xl = yp, yl = 1 - xp)
    val det = YoloSegDecoder.Detection(41, 0.9f, 0f, 0f, 0.5f, 1f, BooleanArray(4) { true }, 2, 2)
    val lab = instanceLabels(listOf(det), intArrayOf(7), 8, 6)
    check(lab[5 * 8 + 3] == 7.toShort() && lab[0 * 8 + 3] == 0.toShort(), "portrait mask -> landscape labels")
    check(portraitToSensor(0.25f, 0.5f) == Pair(0.5f, 0.75f), "tap position -> sensor coordinates")

    // fixture scan for the Python loader
    val out = File(args.getOrElse(0) { "fixture" }).apply { mkdirs() }
    File(out, "tracks.json").writeText(ObjectJson.tracks(tr.confirmed()))
    val sel = listOf(
        SelectedObject(1, "bottle", "water \"bottle\"", CaptureMode.PICKABLE, "detected"),
        SelectedObject(101, "cupboard", "cupboard", CaptureMode.ARTICULATED, "tap", 57, 0.41f, 0.63f),
    )
    sel[0].captures["scan"] = "captures/1/scan"
    sel[1].captures["closed"] = "captures/101/closed"
    sel[1].captures["open"] = "captures/101/open"
    File(out, "selection.json").writeText(ObjectJson.selection(sel))
    println("fixture written to $out")
}
