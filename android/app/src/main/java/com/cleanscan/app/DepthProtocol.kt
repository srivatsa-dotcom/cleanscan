package com.cleanscan.app

import java.nio.ByteBuffer
import java.nio.ByteOrder

/*
 * Depth-module protocol (rig/cleanscan_rig/protocol.py, module/firmware): CSD1 frame
 * reassembly and the ping/pong clock sync. No Android dependencies: tested on the JVM by
 * tools/app-logic-test.
 */

class ModuleDepthFrame(val frameId: Long, val t: Double, val width: Int, val height: Int, val mm: ShortArray)

/** Reassembles CSD1 frames from chunks in any order; frames with lost chunks are dropped. */
class DepthFrameAssembler(private val keep: Int = 4) {
    private class Parts(val n: Int, val t: Double, val w: Int, val h: Int) {
        val chunks = HashMap<Int, ByteArray>()
    }

    private val parts = java.util.TreeMap<Long, Parts>()
    var dropped = 0
        private set

    /** Returns a frame when [packet] completes one, else null. Throws on non-CSD1 data. */
    fun feed(packet: ByteArray, length: Int = packet.size): ModuleDepthFrame? {
        if (length < HEADER || packet[0] != 'C'.code.toByte() || packet[1] != 'S'.code.toByte() ||
            packet[2] != 'D'.code.toByte() || packet[3] != '1'.code.toByte()) {
            throw IllegalArgumentException("not a CSD1 packet")
        }
        val b = ByteBuffer.wrap(packet, 0, length).order(ByteOrder.LITTLE_ENDIAN)
        if (b.get(4).toInt() != 3 || b.get(5).toInt() != 1) throw IllegalArgumentException("unsupported depth format")
        val w = b.getShort(6).toInt() and 0xFFFF
        val h = b.getShort(8).toInt() and 0xFFFF
        val fid = b.getInt(10).toLong() and 0xFFFFFFFFL
        val k = b.getShort(14).toInt() and 0xFFFF
        val n = b.getShort(16).toInt() and 0xFFFF
        val t = b.getDouble(18)
        val p = parts.getOrPut(fid) { Parts(n, t, w, h) }
        p.chunks[k] = packet.copyOfRange(HEADER, length)
        while (parts.size > keep) {
            val oldest = parts.firstKey()
            if (oldest == fid) break
            parts.remove(oldest)
            dropped++
        }
        if (p.chunks.size < p.n) return null
        val older = parts.headMap(fid).size
        dropped += older
        parts.headMap(fid).clear()
        parts.remove(fid)
        val total = (0 until p.n).sumOf { p.chunks[it]?.size ?: 0 }
        if (total != 2 * p.w * p.h) { dropped++; return null }
        val bytes = ByteBuffer.allocate(total).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until p.n) bytes.put(p.chunks[i]!!)
        bytes.flip()
        val mm = ShortArray(p.w * p.h)
        bytes.asShortBuffer().get(mm)
        return ModuleDepthFrame(fid, p.t, p.w, p.h, mm)
    }

    companion object {
        const val HEADER = 26  // "<4sBBHHIHHd"
    }
}

/** Module clock -> phone clock from ping/pong (the exchange with the shortest round trip). */
class ClockSync(private val keep: Int = 30) {
    private val samples = ArrayDeque<Pair<Double, Double>>()  // (rtt, offset = module - phone)

    fun add(t1: Double, t2: Double, t3: Double) {
        samples.addLast(Pair(t3 - t1, t2 - (t1 + t3) / 2))
        while (samples.size > keep) samples.removeFirst()
    }

    val ready get() = samples.isNotEmpty()
    val offset get() = samples.minByOrNull { it.first }!!.second
    val rtt get() = samples.minByOrNull { it.first }!!.first
    fun toPhone(moduleTime: Double) = moduleTime - offset
}

/** Pinhole intrinsics (fx, fy, cx, cy) of a module image from its field of view. */
fun moduleIntrinsics(width: Int, height: Int, fovH: Double, fovV: Double): DoubleArray = doubleArrayOf(
    width / 2.0 / Math.tan(Math.toRadians(fovH) / 2), height / 2.0 / Math.tan(Math.toRadians(fovV) / 2),
    width / 2.0, height / 2.0)
