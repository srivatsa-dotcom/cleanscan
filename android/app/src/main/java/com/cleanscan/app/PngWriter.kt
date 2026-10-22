package com.cleanscan.app

import java.io.ByteArrayOutputStream
import java.io.DataOutputStream
import java.io.File
import java.util.zip.CRC32
import java.util.zip.Deflater
import java.util.zip.DeflaterOutputStream

/** Minimal grayscale PNG encoder (8 or 16 bit) - Android's Bitmap has no 16-bit gray. */
object PngWriter {
    fun writeGray8(file: File, width: Int, height: Int, px: ByteArray) =
        write(file, width, height, 8) { out, y -> out.write(px, y * width, width) }

    fun writeGray16(file: File, width: Int, height: Int, px: ShortArray) =
        write(file, width, height, 16) { out, y ->
            val row = ByteArray(width * 2)
            for (x in 0 until width) {
                val v = px[y * width + x].toInt()
                row[2 * x] = (v shr 8).toByte()  // PNG is big-endian
                row[2 * x + 1] = v.toByte()
            }
            out.write(row)
        }

    private fun write(file: File, width: Int, height: Int, bits: Int,
                      row: (DeflaterOutputStream, Int) -> Unit) {
        val raw = ByteArrayOutputStream()
        DeflaterOutputStream(raw, Deflater(6)).use { z ->
            for (y in 0 until height) {
                z.write(0)  // filter: none
                row(z, y)
            }
        }
        DataOutputStream(file.outputStream().buffered()).use { out ->
            out.write(byteArrayOf(0x89.toByte(), 'P'.code.toByte(), 'N'.code.toByte(), 'G'.code.toByte(),
                0x0D, 0x0A, 0x1A, 0x0A))
            val ihdr = ByteArrayOutputStream()
            DataOutputStream(ihdr).apply {
                writeInt(width); writeInt(height)
                writeByte(bits); writeByte(0)  // grayscale
                writeByte(0); writeByte(0); writeByte(0)
            }
            chunk(out, "IHDR", ihdr.toByteArray())
            chunk(out, "IDAT", raw.toByteArray())
            chunk(out, "IEND", ByteArray(0))
        }
    }

    private fun chunk(out: DataOutputStream, type: String, data: ByteArray) {
        val t = type.toByteArray(Charsets.US_ASCII)
        out.writeInt(data.size)
        out.write(t)
        out.write(data)
        val crc = CRC32().apply { update(t); update(data) }
        out.writeInt(crc.value.toInt())
    }
}
