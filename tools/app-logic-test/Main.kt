// JVM checks for the app's object-capture logic (android/.../ObjectCapture.kt), plus a fixture
// scan whose tracks.json / selection.json the Python pipeline must read. Run: ./run.sh <out_dir>
import com.cleanscan.app.Box
import com.cleanscan.app.COCO_NAMES
import com.cleanscan.app.CaptureMode
import com.cleanscan.app.CoverageRing
import com.cleanscan.app.ObjectJson
import com.cleanscan.app.ObjectTracker
import com.cleanscan.app.SelectedObject
import com.cleanscan.app.ClockSync
import com.cleanscan.app.DepthFrameAssembler
import com.cleanscan.app.YoloSegDecoder
import com.cleanscan.app.moduleIntrinsics
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

    // depth module: CSD1 frames built byte by byte as the firmware sends them
    fun packets(fid: Int, t: Double, w: Int, h: Int, mm: ShortArray): List<ByteArray> {
        val data = java.nio.ByteBuffer.allocate(2 * w * h).order(java.nio.ByteOrder.LITTLE_ENDIAN)
        mm.forEach { data.putShort(it) }
        val bytes = data.array()
        val per = (1400 - 26) and 1.inv()
        val n = (bytes.size + per - 1) / per
        return (0 until n).map { k ->
            val chunk = bytes.copyOfRange(k * per, minOf(bytes.size, (k + 1) * per))
            java.nio.ByteBuffer.allocate(26 + chunk.size).order(java.nio.ByteOrder.LITTLE_ENDIAN)
                .put("CSD1".toByteArray()).put(3).put(1).putShort(w.toShort()).putShort(h.toShort())
                .putInt(fid).putShort(k.toShort()).putShort(n.toShort()).putDouble(t).put(chunk).array()
        }
    }
    val img = ShortArray(100 * 100) { ((it * 37) % 2600).toShort() }
    val pk = packets(42, 123.5, 100, 100, img).shuffled(java.util.Random(1))
    val asm = DepthFrameAssembler()
    val got = pk.mapNotNull { asm.feed(it) }
    check(pk.size == 15 && got.size == 1 && got[0].mm.contentEquals(img) && got[0].t == 123.5 && got[0].frameId == 42L,
        "depth frame reassembled from 15 shuffled packets")
    packets(43, 124.0, 100, 100, img).dropLast(1).forEach { asm.feed(it) }
    val next = packets(44, 124.1, 100, 100, img).mapNotNull { asm.feed(it) }
    check(next.single().frameId == 44L && asm.dropped == 1, "a frame with a lost packet is dropped")
    val cs = ClockSync()
    cs.add(10.0, 1010.02, 10.05)  // slow exchange
    cs.add(11.0, 1011.0025, 11.005)  // fast one wins: offset = 1011.0025 - 11.0025 = 1000
    check(kotlin.math.abs(cs.offset - 1000.0) < 1e-9 && kotlin.math.abs(cs.toPhone(1500.0) - 500.0) < 1e-9,
        "clock sync uses the shortest round trip")
    val k = moduleIntrinsics(100, 100, 70.0, 60.0)
    check(kotlin.math.abs(k[0] - 71.41) < 0.01 && k[2] == 50.0, "A010 intrinsics from its field of view")

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
