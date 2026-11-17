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
    private var roomWriter: SessionWriter? = null
    private lateinit var flow: GuidedCaptureFlow
    private lateinit var depthClient: DepthModuleClient
    private lateinit var depth: DepthRecording
    private val prefs by lazy { getSharedPreferences("cleanscan", MODE_PRIVATE) }
    private var clockChecked = false
    private val ui = android.os.Handler(android.os.Looper.getMainLooper())
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
        val scansButton = findViewById<Button>(R.id.scans)
        depthClient = DepthModuleClient(this)
        depth = DepthRecording(depthClient) { prefs.getBoolean("depthModule", false) }
        if (prefs.getBoolean("depthModule", false)) depthClient.start(prefs.getString("depthModuleHost", "") ?: "")
        ui.postDelayed(object : Runnable {
            override fun run() {
                if (recorder == null && !flow.active && prefs.getBoolean("depthModule", false)) {
                    status.text = if (depthClient.connected)
                        "Depth module: ${depthClient.sensor.uppercase()} · ${depthClient.framesPerSecond} fps" +
                            (depthClient.rttMs?.let { " · sync %.0f ms".format(it) } ?: "")
                    else "Depth module: looking for it - join the CleanScanDepth Wi-Fi"
                }
                ui.postDelayed(this, 2000)
            }
        }, 2000)
        flow = GuidedCaptureFlow(this, segmenter, status, recordButton, depth) { scansButton.isEnabled = true }
        recordButton.setOnClickListener { if (flow.active) flow.toggle() else toggleRecording() }
        scansButton.setOnClickListener { showScans() }
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
        if (flow.active && flow.recorder != null) flow.toggle()
        surface.onPause()
        session?.pause()
    }

    override fun onDestroy() {
        depthClient.stop()
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
            roomWriter = writer
            depth.start(writer.dir)
            // object detection + tracking needs the YOLO model; without it objects are added by tapping
            recorder = ScanRecorder(writer, segmenter,
                onFrameWritten = { n, masked, mask ->
                    runOnUiThread {
                        status.text = "Recording · $n frames · %.1f%% living beings masked".format(masked * 100)
                        maskPreview.setImageBitmap(mask)
                    }
                },
                onError = { msg -> runOnUiThread { status.text = msg } },
                tracker = ObjectTracker())
            recordButton.text = "Stop"
            maskPreview.visibility = View.VISIBLE
            status.text = "Recording - move slowly, cover every wall and the floor"
        } else {
            recorder = null
            recordButton.text = "Record"
            maskPreview.visibility = View.GONE
            val dir = roomWriter?.dir
            depth.finish()
            findViewById<Button>(R.id.scans).isEnabled = false
            current.finish {
                runOnUiThread {
                    status.text = "Saved the room. Choose interactive objects."
                    if (dir != null) flow.start(dir, current.tracker!!, current.keyframes.toList())
                }
            }
        }
    }

    private fun showScans() {
        val scans = SessionWriter.list(this)
        val labels = scans.map { "${it.name}  (${SessionWriter.frameCount(it)} frames)" }.toTypedArray()
        val on = prefs.getBoolean("depthModule", false)
        AlertDialog.Builder(this)
            .setTitle(if (scans.isEmpty()) "No scans yet" else "Export a scan")
            .setItems(labels) { _, i -> share(scans[i]) }
            .setNeutralButton(if (on) "Depth module: on" else "Depth module: off") { _, _ -> toggleDepthModule() }
            .setNegativeButton("Close", null)
            .show()
    }

    /** External depth module (module/README.md): clip-on ToF camera for phones without a ToF sensor. */
    private fun toggleDepthModule() {
        val on = !prefs.getBoolean("depthModule", false)
        prefs.edit().putBoolean("depthModule", on).apply()
        if (on) depthClient.start(prefs.getString("depthModuleHost", "") ?: "") else depthClient.stop()
        Toast.makeText(this, if (on) "Depth module on: join the CleanScanDepth Wi-Fi" else "Depth module off",
            Toast.LENGTH_LONG).show()
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
        if (!clockChecked && frame.timestamp > 0) {
            // which clock ARCore's timestamps use: boot time on most phones, monotonic on some
            clockChecked = true
            val boot = android.os.SystemClock.elapsedRealtimeNanos()
            depthClient.useBootTime = kotlin.math.abs(frame.timestamp - boot) < 1_000_000_000L
        }
        background.draw(frame)
        recorder?.onFrame(frame)
        flow.onFrame(frame, viewW, viewH)
    }
}
