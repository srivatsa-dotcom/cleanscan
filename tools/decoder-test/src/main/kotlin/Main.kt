import com.cleanscan.app.YoloSegDecoder
import java.io.DataInputStream
import java.io.DataOutputStream
import java.io.File

fun readInts(f: File): IntArray = DataInputStream(f.inputStream().buffered()).use { s ->
    IntArray((f.length() / 4).toInt()) { s.readInt() } }
fun readFloats(f: File): FloatArray = DataInputStream(f.inputStream().buffered()).use { s ->
    FloatArray((f.length() / 4).toInt()) { s.readFloat() } }
fun writeFloats(f: File, a: FloatArray) = DataOutputStream(f.outputStream().buffered()).use { o -> a.forEach { o.writeFloat(it) } }
fun shape(s: String) = s.split(",").map { it.toInt() }.toIntArray()

// usage: pack <argb.bin> <inputShape> <out.bin>
//        decode <det.bin> <detShape> <proto.bin> <protoShape> <inputShape> <depthW> <depthH> <mask.bin>
fun main(a: Array<String>) {
    when (a[0]) {
        "pack" -> {
            val layout = YoloSegDecoder.InputLayout.of(shape(a[2]))
            writeFloats(File(a[3]), YoloSegDecoder.packInput(readInts(File(a[1])), layout))
            println("packed ${layout.width}x${layout.height} channelsFirst=${layout.channelsFirst}")
        }
        "decode" -> {
            val out = YoloSegDecoder.Outputs.of(shape(a[2]), readFloats(File(a[1])), shape(a[4]), readFloats(File(a[3])))
            val layout = YoloSegDecoder.InputLayout.of(shape(a[5]))
            val dets = YoloSegDecoder.decode(out.det, out.channels, out.anchors, out.proto, out.numMasks,
                out.maskH, out.maskW, layout.width, layout.height)
            val w = a[6].toInt(); val h = a[7].toInt()
            val mask = ByteArray(w * h)
            YoloSegDecoder.rasterize(dets, w, h, mask)
            File(a[8]).writeBytes(mask)
            println("detections: " + dets.joinToString { "cls=${it.classId}(${"%.2f".format(it.score)})" })
        }
    }
}
