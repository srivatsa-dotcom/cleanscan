package com.cleanscan.app

import android.annotation.SuppressLint
import android.graphics.BitmapFactory
import android.graphics.Color
import android.view.MotionEvent
import android.view.View
import android.widget.ArrayAdapter
import android.widget.Button
import android.widget.CheckBox
import android.widget.EditText
import android.widget.HorizontalScrollView
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.ProgressBar
import android.widget.ScrollView
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import com.google.ar.core.Frame
import com.google.ar.core.Plane
import com.google.ar.core.Point
import com.google.ar.core.TrackingState
import java.io.File

/**
 * After the room pass: "Which objects should be interactive for RL?", then one guided capture
 * per selected object, all in the same ARCore session (one world frame for everything).
 *
 *   pickable     put it on a flat surface, walk around it (coverage ring, done at >= 80 %)
 *   articulated  scan it closed, then open it fully and scan again
 *   transparent  side profile with the phone level, then one shot from the top
 *
 * Writes captures/<id>/<stage>/ sessions and selection.json into the scan folder
 * (docs/capture_format.md). Logic without Android dependencies is in ObjectCapture.kt.
 */
class GuidedCaptureFlow(
    private val activity: AppCompatActivity,
    private val segmenter: LivingSegmenter,
    private val status: TextView,
    private val recordButton: Button,
    private val depth: DepthRecording,
    private val onFinished: () -> Unit,
) {
    private class Candidate(val obj: SelectedObject, var checked: Boolean, val thumb: File?, val frames: Int)

    private val guide: View = activity.findViewById(R.id.guide)
    private val title: TextView = activity.findViewById(R.id.guideTitle)
    private val text: TextView = activity.findViewById(R.id.guideText)
    private val progress: ProgressBar = activity.findViewById(R.id.guideProgress)
    private val meter: TextView = activity.findViewById(R.id.guideMeter)

    private var scanDir: File? = null
    private val candidates = ArrayList<Candidate>()
    private var keyframes: List<ScanRecorder.Keyframe> = emptyList()
    private val selected = ArrayList<SelectedObject>()
    private var current = 0
    private var stage = 0
    private var nextTapId = 101

    @Volatile var recorder: ScanRecorder? = null
        private set
    @Volatile var active = false
        private set
    @Volatile private var coverage: CoverageRing? = null
    @Volatile private var center: FloatArray? = null
    @Volatile private var needCenter = false
    @Volatile private var pitch = 0f
    private var lastUi = 0L

    init {
        activity.findViewById<Button>(R.id.guideSkip).setOnClickListener { skipObject() }
    }

    // ------------------------------------------------------------------ selection

    /** Called when the room pass is saved. */
    fun start(scanDir: File, tracker: ObjectTracker, keyframes: List<ScanRecorder.Keyframe>) {
        this.scanDir = scanDir
        this.keyframes = keyframes
        candidates.clear()
        for (t in tracker.confirmed()) {
            val thumb = File(scanDir, "thumbs/${t.id}.jpg").takeIf { it.exists() }
            candidates.add(Candidate(SelectedObject(t.id, t.className, t.className, CaptureMode.forClass(t.className),
                "detected"), false, thumb, t.frames))
        }
        File(scanDir, "tracks.json").writeText(ObjectJson.tracks(tracker.confirmed()))
        active = true
        showSelection()
    }

    private fun summary(): String {
        if (candidates.isEmpty()) return "No objects were detected. Add objects by tapping them on a photo."
        val counts = candidates.groupingBy { it.obj.cls }.eachCount()
        return "Found: " + counts.entries.joinToString(", ") { (c, n) -> if (n == 1) c else "$n× $c" } +
            ". Which should be interactive for RL?"
    }

    private fun showSelection() {
        val ctx = activity
        val list = LinearLayout(ctx).apply { orientation = LinearLayout.VERTICAL; setPadding(32, 16, 32, 0) }
        list.addView(TextView(ctx).apply { text = summary(); setPadding(0, 0, 0, 16) })
        for (cls in candidates.map { it.obj.cls }.distinct()) {
            list.addView(TextView(ctx).apply { text = cls.uppercase(); setTextColor(Color.GRAY); setPadding(0, 16, 0, 4) })
            for (c in candidates.filter { it.obj.cls == cls }) list.addView(row(c))
        }
        AlertDialog.Builder(ctx)
            .setTitle("Interactive objects")
            .setView(ScrollView(ctx).apply { addView(list) })
            .setCancelable(false)
            .setPositiveButton("Scan selected") { _, _ -> beginCaptures() }
            .setNeutralButton("Add object…") { _, _ -> showAddObject() }
            .setNegativeButton("Room only") { _, _ -> finish() }
            .show()
    }

    private fun row(c: Candidate): View {
        val ctx = activity
        return LinearLayout(ctx).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(0, 6, 0, 6)
            val img = ImageView(ctx).apply {
                layoutParams = LinearLayout.LayoutParams(140, 140)
                scaleType = ImageView.ScaleType.CENTER_CROP
                c.thumb?.let { setImageBitmap(BitmapFactory.decodeFile(it.path)) }
                setBackgroundColor(Color.DKGRAY)
            }
            addView(img)
            val col = LinearLayout(ctx).apply {
                orientation = LinearLayout.VERTICAL
                layoutParams = LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f)
                setPadding(16, 0, 0, 0)
            }
            col.addView(CheckBox(ctx).apply {
                text = if (c.obj.source == "tap") "${c.obj.name} (added)" else "${c.obj.name} #${c.obj.id} · ${c.frames} frames"
                isChecked = c.checked
                setOnCheckedChangeListener { _, v -> c.checked = v }
            })
            col.addView(modeSpinner(c.obj.mode) { c.obj.mode = it })
            addView(col)
        }
    }

    private fun modeSpinner(initial: CaptureMode, onChange: (CaptureMode) -> Unit): Spinner {
        val modes = CaptureMode.values()
        return Spinner(activity).apply {
            adapter = ArrayAdapter(activity, android.R.layout.simple_spinner_dropdown_item,
                modes.map { "${it.label} (${it.key})" })
            setSelection(modes.indexOf(initial))
            onItemSelectedListener = object : android.widget.AdapterView.OnItemSelectedListener {
                override fun onItemSelected(p: android.widget.AdapterView<*>?, v: View?, pos: Int, id: Long) = onChange(modes[pos])
                override fun onNothingSelected(p: android.widget.AdapterView<*>?) {}
            }
        }
    }

    /** Tap an object that was not detected on one of the room-pass photos, then name it. */
    @SuppressLint("ClickableViewAccessibility")
    private fun showAddObject() {
        val ctx = activity
        if (keyframes.isEmpty()) {
            Toast.makeText(ctx, "No photos from the room pass to tap on", Toast.LENGTH_SHORT).show()
            showSelection()
            return
        }
        lateinit var dialog: AlertDialog
        val strip = LinearLayout(ctx).apply { orientation = LinearLayout.HORIZONTAL }
        for (kf in keyframes) {
            strip.addView(ImageView(ctx).apply {
                setImageBitmap(kf.bitmap)
                adjustViewBounds = true
                layoutParams = LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, 900).apply { marginEnd = 12 }
                setOnTouchListener { v, ev ->
                    if (ev.action == MotionEvent.ACTION_UP) {
                        val (u, w) = portraitToSensor(ev.x / v.width, ev.y / v.height)
                        dialog.dismiss()
                        nameNewObject(kf.index, u, w)
                    }
                    true
                }
            })
        }
        dialog = AlertDialog.Builder(ctx)
            .setTitle("Tap the object on a photo")
            .setView(HorizontalScrollView(ctx).apply { addView(strip) })
            .setNegativeButton("Back") { _, _ -> showSelection() }
            .setCancelable(false)
            .show()
    }

    private fun nameNewObject(frame: Int, u: Float, v: Float) {
        val ctx = activity
        val name = EditText(ctx).apply { hint = "e.g. cupboard, pressure cooker" }
        var mode = CaptureMode.PICKABLE
        val spinner = modeSpinner(mode) { mode = it }
        name.addTextChangedListener(object : android.text.TextWatcher {
            override fun afterTextChanged(s: android.text.Editable?) {
                spinner.setSelection(CaptureMode.values().indexOf(CaptureMode.forClass(s.toString())))
            }
            override fun beforeTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
            override fun onTextChanged(s: CharSequence?, a: Int, b: Int, c: Int) {}
        })
        val box = LinearLayout(ctx).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(48, 16, 48, 0)
            addView(name)
            addView(spinner)
        }
        AlertDialog.Builder(ctx)
            .setTitle("What is it?")
            .setView(box)
            .setCancelable(false)
            .setPositiveButton("Add") { _, _ ->
                val label = name.text.toString().trim().ifEmpty { "object" }
                candidates.add(Candidate(SelectedObject(nextTapId++, label.lowercase(), label, mode, "tap", frame, u, v),
                    true, null, 0))
                showSelection()
            }
            .setNegativeButton("Cancel") { _, _ -> showSelection() }
            .show()
    }

    // ------------------------------------------------------------------ guided capture

    private fun beginCaptures() {
        selected.clear()
        selected.addAll(candidates.filter { it.checked }.map { it.obj })
        if (selected.isEmpty()) return finish()
        current = 0
        stage = 0
        showStage()
    }

    private val obj get() = selected[current]
    private val stageName get() = obj.mode.stages[stage]

    private fun instructions(): Pair<String, String> {
        val n = obj.name
        return when (obj.mode to stageName) {
            CaptureMode.PICKABLE to "scan" -> "Scan the $n (${current + 1}/${selected.size})" to
                "Put it on a flat surface with space around it, point the phone at it, press Record and walk slowly all the way around it."
            CaptureMode.ARTICULATED to "closed" -> "Scan the $n closed (${current + 1}/${selected.size})" to
                "Close it. Stand 1.5–2 m away so all of it is in view, press Record and sweep slowly from side to side."
            CaptureMode.ARTICULATED to "open" -> "Now open the $n fully" to
                "Open the door / pull out the drawer as far as it goes. Press Record and scan it again the same way."
            CaptureMode.TRANSPARENT to "profile" -> "Side profile of the $n (${current + 1}/${selected.size})" to
                "Hold the phone level at the height of the $n, about 30 cm away. Press Record and move sideways around its front."
            else -> "Top view of the $n" to
                "Hold the phone flat above the $n, screen up, so it is in the middle of the picture. Press Record for a second."
        }
    }

    private fun showStage() {
        val (t, s) = instructions()
        title.text = t
        text.text = s
        progress.visibility = if (obj.mode == CaptureMode.PICKABLE) View.VISIBLE else View.GONE
        progress.progress = 0
        meter.visibility = if (obj.mode == CaptureMode.TRANSPARENT) View.VISIBLE else View.GONE
        guide.visibility = View.VISIBLE
        recordButton.text = "Record"
        status.text = "Guided capture"
    }

    /** Record button while the flow is active. */
    fun toggle() {
        if (recorder == null) startStage() else stopStage()
    }

    private fun startStage() {
        val dir = scanDir ?: return
        val rel = "captures/${obj.id}/$stageName"
        val writer = SessionWriter.createCapture(dir, rel)
        depth.start(writer.dir)
        obj.captures[stageName] = rel
        val cocoIndex = COCO_NAMES.indexOf(obj.cls)
        val captureClass = when {
            obj.mode == CaptureMode.ARTICULATED -> null  // the pipeline finds the moving part itself
            cocoIndex >= 0 -> cocoIndex
            else -> -1  // not a detector class: the most central detection
        }
        val gate: ((FloatArray) -> Boolean)? = when (obj.mode to stageName) {
            CaptureMode.TRANSPARENT to "profile" -> { p -> kotlin.math.abs(cameraPitchDeg(p)) < 15f }
            CaptureMode.TRANSPARENT to "top" -> { p -> cameraPitchDeg(p) < -65f }
            else -> null
        }
        coverage = if (obj.mode == CaptureMode.PICKABLE) CoverageRing() else null
        center = null
        needCenter = obj.mode == CaptureMode.PICKABLE
        recorder = ScanRecorder(writer, segmenter,
            onFrameWritten = { n, _, _ -> activity.runOnUiThread { status.text = "Recording ${obj.name} · $n frames" } },
            onError = { msg -> activity.runOnUiThread { status.text = msg } },
            captureClass = captureClass, gate = gate)
        recordButton.text = "Done"
    }

    private fun stopStage() {
        val r = recorder ?: return
        recorder = null
        depth.finish()
        val cov = coverage?.fraction
        r.finish {}
        if (cov != null && cov < 0.8f) {
            Toast.makeText(activity, "Only ${(cov * 100).toInt()} % of the way around was covered", Toast.LENGTH_LONG).show()
        }
        coverage = null
        stage++
        if (stage >= obj.mode.stages.size) {
            stage = 0
            current++
        }
        if (current >= selected.size) finish() else showStage()
    }

    private fun skipObject() {
        recorder?.let { it.finish {}; recorder = null; depth.finish() }
        obj.captures.clear()
        stage = 0
        current++
        if (current >= selected.size) finish() else showStage()
    }

    private fun finish() {
        scanDir?.let { File(it, "selection.json").writeText(ObjectJson.selection(selected)) }
        active = false
        guide.visibility = View.GONE
        recordButton.text = "Record"
        status.text = if (selected.isEmpty()) "Saved (room only). Tap Scans to export."
        else "Saved with ${selected.size} interactive objects. Tap Scans to export."
        onFinished()
    }

    // ------------------------------------------------------------------ per AR frame (GL thread)

    fun onFrame(frame: Frame, viewW: Int, viewH: Int) {
        if (!active) return
        val cam = frame.camera
        if (cam.trackingState != TrackingState.TRACKING) return
        val col = FloatArray(16).also { cam.pose.toMatrix(it, 0) }
        val row = FloatArray(16)
        for (r in 0 until 4) for (c in 0 until 4) row[r * 4 + c] = col[c * 4 + r]
        pitch = cameraPitchDeg(row)
        recorder?.onFrame(frame)
        if (needCenter && viewW > 0) {
            // the object the user points at: a hit on a plane or feature point at the screen centre,
            // else 50 cm in front of the camera
            val hit = frame.hitTest(viewW / 2f, viewH / 2f).firstOrNull {
                val t = it.trackable
                (t is Plane && t.isPoseInPolygon(it.hitPose)) || t is Point
            }
            center = hit?.hitPose?.let { floatArrayOf(it.tx(), it.ty(), it.tz()) }
                ?: floatArrayOf(cam.pose.tx() - 0.5f * row[2], cam.pose.ty() - 0.5f * row[6], cam.pose.tz() - 0.5f * row[10])
            needCenter = false
        }
        val ring = coverage
        val c = center
        if (ring != null && c != null && recorder != null) ring.add(cam.pose.tx(), cam.pose.tz(), c[0], c[2])
        val now = System.currentTimeMillis()
        if (now - lastUi > 150) {
            lastUi = now
            val cov = ring?.fraction
            val p = pitch
            val mode = if (current < selected.size) obj.mode else null
            val st = if (current < selected.size) stageName else ""
            activity.runOnUiThread {
                if (cov != null) {
                    progress.progress = (cov * 100).toInt()
                    if (cov >= 0.8f) text.text = "Covered ${(cov * 100).toInt()} % - press Done."
                }
                if (mode == CaptureMode.TRANSPARENT) {
                    val ok = if (st == "top") p < -65f else kotlin.math.abs(p) < 15f
                    meter.text = (if (ok) "✓ " else "✗ ") + "phone tilt %.0f°".format(p) +
                        (if (st == "top") " (point it straight down)" else " (hold it level)")
                    meter.setTextColor(if (ok) Color.GREEN else Color.YELLOW)
                }
            }
        }
    }
}
