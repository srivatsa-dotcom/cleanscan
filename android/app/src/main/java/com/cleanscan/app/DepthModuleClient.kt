package com.cleanscan.app

import android.content.Context
import android.net.wifi.WifiManager
import android.os.SystemClock
import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.io.FileWriter
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetSocketAddress
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * Talks to the clip-on depth module (module/: ESP32-S3 + A010, or a Pi + Arducam ToF):
 * discovers it (UDP broadcast + the usual addresses), keeps its clock in sync with the phone's
 * and delivers reassembled depth frames on the phone clock (protocol: rig/cleanscan_rig/protocol.py).
 *
 * The phone clock is the one ARCore's frame timestamps use: [useBootTime] is set from the first
 * frame (boot time on most phones, the monotonic clock on some).
 */
class DepthModuleClient(context: Context) {
    @Volatile var connected = false; private set
    @Volatile var sensor = ""; private set
    @Volatile var host = ""; private set
    @Volatile var framesPerSecond = 0; private set
    @Volatile var fovDeg = doubleArrayOf(70.0, 60.0); private set
    @Volatile var useBootTime = true
    @Volatile var onFrame: ((ModuleDepthFrame) -> Unit)? = null

    val rttMs: Double? get() = if (sync.ready) sync.rtt * 1000 else null

    private val wifi = context.applicationContext.getSystemService(Context.WIFI_SERVICE) as WifiManager
    private var lock: WifiManager.MulticastLock? = null
    private var socket: DatagramSocket? = null
    private var target: InetSocketAddress? = null
    private val sync = ClockSync()
    private val assembler = DepthFrameAssembler()
    private val timer = Executors.newSingleThreadScheduledExecutor()
    private var tick: java.util.concurrent.ScheduledFuture<*>? = null
    @Volatile private var lastSeen = 0.0
    @Volatile private var counted = 0
    private var pingId = 0

    fun now(): Double = (if (useBootTime) SystemClock.elapsedRealtimeNanos() else System.nanoTime()) / 1e9

    fun start(preferredHost: String) {
        stop()
        lock = wifi.createMulticastLock("cleanscan-depth").apply { setReferenceCounted(false); acquire() }
        val s = DatagramSocket(null).apply { reuseAddress = true; broadcast = true; bind(InetSocketAddress(0)) }
        socket = s
        val hosts = (listOf(preferredHost, "255.255.255.255") + DEFAULT_HOSTS).filter { it.isNotBlank() }.distinct()
        thread(name = "depth-module-rx", isDaemon = true) { receive(s) }
        tick = timer.scheduleWithFixedDelay({
            try {
                val t = target
                if (t == null) hosts.forEach { send(s, JSONObject().put("op", "hello"), InetSocketAddress(it, PORT)) }
                else {
                    send(s, JSONObject().put("op", "hello"), t)
                    send(s, JSONObject().put("op", "ping").put("id", ++pingId).put("t1", now()), t)
                }
                framesPerSecond = counted * 2
                counted = 0
                connected = now() - lastSeen < 3
            } catch (e: Exception) {
                Log.w(TAG, "tick: ${e.message}")
            }
        }, 0, 500, java.util.concurrent.TimeUnit.MILLISECONDS)
    }

    fun stop() {
        tick?.cancel(false)
        socket?.close()
        socket = null
        target = null
        lock?.release()
        lock = null
        connected = false
    }

    private fun send(s: DatagramSocket, obj: JSONObject, to: InetSocketAddress) {
        val b = obj.toString().toByteArray()
        try { s.send(DatagramPacket(b, b.size, to)) } catch (_: Exception) {}
    }

    private fun receive(s: DatagramSocket) {
        val buf = ByteArray(2048)
        while (!s.isClosed) {
            val p = DatagramPacket(buf, buf.size)
            try { s.receive(p) } catch (_: Exception) { return }
            val t3 = now()
            if (p.length > 0 && buf[0] == '{'.code.toByte()) {
                val msg = try { JSONObject(String(buf, 0, p.length)) } catch (_: Exception) { continue }
                when (msg.optString("op")) {
                    "info" -> if (msg.optString("kind") == "depth") {
                        if (target == null) target = InetSocketAddress(p.address, p.port)  // first module wins
                        if (InetSocketAddress(p.address, p.port) == target) {
                            lastSeen = t3
                            sensor = msg.optString("sensor")
                            host = p.address.hostAddress ?: ""
                            msg.optJSONArray("fov_deg")?.let { fovDeg = doubleArrayOf(it.getDouble(0), it.getDouble(1)) }
                        }
                    }
                    "pong" -> sync.add(msg.getDouble("t1"), msg.getDouble("t2"), t3)
                }
                continue
            }
            if (InetSocketAddress(p.address, p.port) != target) continue
            val f = try { assembler.feed(buf, p.length) } catch (_: Exception) { null } ?: continue
            if (!sync.ready) continue
            lastSeen = t3
            counted++
            onFrame?.invoke(ModuleDepthFrame(f.frameId, sync.toPhone(f.t), f.width, f.height, f.mm))
        }
    }

    companion object {
        private const val TAG = "DepthModule"
        const val PORT = 5600
        val DEFAULT_HOSTS = listOf("192.168.4.1", "10.42.0.1")  // ESP32 access point, Pi hotspot
    }
}

/** Writes module frames next to a session: tof/*.png (16-bit mm), tof.jsonl, depth_module.json. */
class DepthRecorder(private val dir: File) {
    private val worker = Executors.newSingleThreadExecutor()
    private val lines = FileWriter(File(dir, "tof.jsonl").also { File(dir, "tof").mkdirs() }, true)
    private var index = 0

    fun add(f: ModuleDepthFrame, fovDeg: DoubleArray) = worker.execute {
        val name = "%06d".format(index)
        PngWriter.writeGray16(File(dir, "tof/$name.png"), f.width, f.height, f.mm)
        val k = moduleIntrinsics(f.width, f.height, fovDeg[0], fovDeg[1])
        lines.write(JSONObject().put("index", index).put("t", f.t).put("depth", "tof/$name.png")
            .put("intrinsics", JSONArray(k.toList())).put("size", JSONArray(listOf(f.width, f.height)))
            .toString() + "\n")
        index++
    }

    fun finish(sensor: String, fovDeg: DoubleArray, rttMs: Double?) {
        worker.execute {
            lines.close()
            File(dir, "depth_module.json").writeText(JSONObject().put("sensor", sensor)
                .put("fov_deg", JSONArray(fovDeg.toList())).put("depth_unit", 0.001).put("frame", "opencv")
                .put("clock_rtt_ms", rttMs ?: -1.0).put("frames", index).toString(2))
        }
        worker.shutdown()
    }
}

/** One recording at a time (room pass or a guided capture) into the session's folder. */
class DepthRecording(private val client: DepthModuleClient, private val enabled: () -> Boolean) {
    private var rec: DepthRecorder? = null

    fun start(dir: File) {
        if (!enabled()) return
        val r = DepthRecorder(dir)
        rec = r
        client.onFrame = { r.add(it, client.fovDeg) }
    }

    fun finish() {
        client.onFrame = null
        rec?.finish(client.sensor, client.fovDeg, client.rttMs)
        rec = null
    }
}
