package com.cleanscan.app

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.opengl.GLES20
import android.opengl.GLSurfaceView
import android.os.Bundle
import android.view.View
import android.view.WindowManager
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.google.ar.core.ArCoreApk
import com.google.ar.core.CameraConfig
import com.google.ar.core.CameraConfigFilter
import com.google.ar.core.Config
import com.google.ar.core.Session
import com.google.ar.core.exceptions.CameraNotAvailableException
import com.google.ar.core.exceptions.UnavailableException
import javax.microedition.khronos.egl.EGLConfig
import javax.microedition.khronos.opengles.GL10
import kotlin.concurrent.thread

class MainActivity : AppCompatActivity(), GLSurfaceView.Renderer {
    private lateinit var surface: GLSurfaceView
    private lateinit var status: TextView
    private lateinit var recordButton: Button
    private lateinit var maskPreview: ImageView

    private var session: Session? = null
    private var installRequested = false
    private val background = BackgroundRenderer()
    private lateinit var segmenter: LivingSegmenter
    @Volatile private var recorder: ScanRecorder? = null
    private var displayRotation = 0
    private var viewW = 0
    private var viewH = 0

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        surface = findViewById(R.id.surface)
        status = findViewById(R.id.status)
        recordButton = findViewById(R.id.record)
        maskPreview = findViewById(R.id.maskPreview)

        surface.preserveEGLContextOnPause = true
        surface.setEGLContextClientVersion(2)
        surface.setRenderer(this)
        surface.renderMode = GLSurfaceView.RENDERMODE_CONTINUOUSLY

        segmenter = LivingSegmenter(this)
        if (!segmenter.available) {
            status.text = "Warning: ${LivingSegmenter.MODEL} missing - people/animals will NOT be masked " +
                "on the phone (desktop --resegment and ghost removal still work)"
        }
        recordButton.setOnClickListener { toggleRecording() }
        findViewById<Button>(R.id.scans).setOnClickListener { showScans() }
    }

    override fun onResume() {
        super.onResume()
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 0)
            return
        }
        if (session == null) {
            try {
                if (ArCoreApk.getInstance().requestInstall(this, !installRequested) ==
                    ArCoreApk.InstallStatus.INSTALL_REQUESTED) {
                    installRequested = true
                    return
                }
                session = createSession()
            } catch (e: UnavailableException) {
                status.text = "ARCore unavailable: ${e.message}"
                return
            }
        }
        try {
            session?.resume()
        } catch (e: CameraNotAvailableException) {
            status.text = "Camera not available"
            session = null
            return
        }
        surface.onResume()
    }

    private fun createSession(): Session {
        val s = Session(this)
        // Prefer the highest-resolution CPU image for sharp textures.
        val filter = CameraConfigFilter(s).setFacingDirection(CameraConfig.FacingDirection.BACK)
        s.getSupportedCameraConfigs(filter).maxByOrNull { it.imageSize.width * it.imageSize.height }
            ?.let { s.cameraConfig = it }
        val config = Config(s).apply {
            focusMode = Config.FocusMode.AUTO
            updateMode = Config.UpdateMode.LATEST_CAMERA_IMAGE
            depthMode = when {
                s.isDepthModeSupported(Config.DepthMode.AUTOMATIC) -> Config.DepthMode.AUTOMATIC
                s.isDepthModeSupported(Config.DepthMode.RAW_DEPTH_ONLY) -> Config.DepthMode.RAW_DEPTH_ONLY
                else -> Config.DepthMode.DISABLED
            }
        }
        if (config.depthMode == Config.DepthMode.DISABLED) {
            status.text = "This phone does not support the ARCore Depth API - scanning is not possible"
            recordButton.isEnabled = false
        }
        s.configure(config)
        return s
    }

    override fun onPause() {
        super.onPause()
        if (recorder != null) toggleRecording()
        surface.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        session?.close()
        session = null
        super.onDestroy()
    }

    override fun onRequestPermissionsResult(code: Int, perms: Array<out String>, results: IntArray) {
        super.onRequestPermissionsResult(code, perms, results)
        if (results.firstOrNull() != PackageManager.PERMISSION_GRANTED) {
            Toast.makeText(this, "Camera permission is required", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    // ---------------------------------------------------------------- recording

    private fun toggleRecording() {
        val current = recorder
        if (current == null) {
            val writer = SessionWriter.create(this)
            recorder = ScanRecorder(writer, segmenter,
                onFrameWritten = { n, masked, mask ->
                    runOnUiThread {
                        status.text = "Recording · $n frames · %.1f%% living beings masked".format(masked * 100)
                        maskPreview.setImageBitmap(mask)
                    }
                },
                onError = { msg -> runOnUiThread { status.text = msg } })
            recordButton.text = "Stop"
            maskPreview.visibility = View.VISIBLE
            status.text = "Recording - move slowly, cover every wall and the floor"
        } else {
            recorder = null
            recordButton.text = "Record"
            maskPreview.visibility = View.GONE
            current.finish { runOnUiThread { status.text = "Saved. Tap Scans to export." } }
        }
    }

    private fun showScans() {
        val scans = SessionWriter.list(this)
        if (scans.isEmpty()) {
            Toast.makeText(this, "No scans yet", Toast.LENGTH_SHORT).show()
            return
        }
        val labels = scans.map { "${it.name}  (${SessionWriter.frameCount(it)} frames)" }.toTypedArray()
        AlertDialog.Builder(this)
            .setTitle("Export a scan")
            .setItems(labels) { _, i -> share(scans[i]) }
            .setNegativeButton("Close", null)
            .show()
    }

    private fun share(dir: java.io.File) {
        status.text = "Zipping ${dir.name}..."
        thread {
            val zip = SessionWriter.zip(this, dir)
            val uri = FileProvider.getUriForFile(this, "$packageName.files", zip)
            runOnUiThread {
                status.text = "Process on your computer: cleanscan process ${zip.name} -o my_world"
                startActivity(Intent.createChooser(
                    Intent(Intent.ACTION_SEND).setType("application/zip")
                        .putExtra(Intent.EXTRA_STREAM, uri)
                        .addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION), "Export scan"))
            }
        }
    }

    // ---------------------------------------------------------------- GL renderer

    override fun onSurfaceCreated(gl: GL10?, config: EGLConfig?) {
        GLES20.glClearColor(0f, 0f, 0f, 1f)
        background.createOnGlThread()
    }

    override fun onSurfaceChanged(gl: GL10?, width: Int, height: Int) {
        GLES20.glViewport(0, 0, width, height)
        viewW = width
        viewH = height
        @Suppress("DEPRECATION")
        displayRotation = windowManager.defaultDisplay.rotation
        session?.setDisplayGeometry(displayRotation, width, height)
    }

    override fun onDrawFrame(gl: GL10?) {
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT or GLES20.GL_DEPTH_BUFFER_BIT)
        val s = session ?: return
        s.setCameraTextureName(background.textureId)
        if (viewW > 0) s.setDisplayGeometry(displayRotation, viewW, viewH)
        val frame = try { s.update() } catch (e: Exception) { return }
        background.draw(frame)
        recorder?.onFrame(frame)
    }
}
