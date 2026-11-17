import SwiftUI

/// After the room pass: which detected objects become interactive, plus objects added by
/// tapping a room-pass photo. Then the guided captures start (GuideOverlay).
struct ObjectSelectionView: View {
    @ObservedObject var scan: ScanController

    var body: some View {
        NavigationStack {
            List {
                Section { Text(scan.selectionSummary).font(.callout) }
                ForEach(groups, id: \.self) { cls in
                    Section(cls.capitalized) {
                        ForEach(scan.candidates.filter { $0.cls == cls }) { obj in
                            CandidateRow(obj: obj)
                        }
                    }
                }
                Section {
                    NavigationLink("Add an object the phone missed…") { AddObjectView(scan: scan) }
                }
            }
            .navigationTitle("Interactive objects")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Room only") { scan.roomOnly() } }
                ToolbarItem(placement: .confirmationAction) { Button("Scan selected") { scan.startCaptures() } }
            }
        }
        .interactiveDismissDisabled()
    }

    private var groups: [String] {
        var seen: [String] = []
        for c in scan.candidates where !seen.contains(c.cls) { seen.append(c.cls) }
        return seen
    }
}

private struct CandidateRow: View {
    @ObservedObject var obj: SelectedObject

    var body: some View {
        HStack(spacing: 12) {
            Group {
                if let url = obj.thumbnail, let img = UIImage(contentsOfFile: url.path) {
                    Image(uiImage: img).resizable().scaledToFill()
                } else {
                    Image(systemName: obj.source == "tap" ? "hand.tap" : "cube").font(.title2)
                }
            }
            .frame(width: 56, height: 56)
            .background(Color.secondary.opacity(0.2))
            .clipShape(RoundedRectangle(cornerRadius: 8))
            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: $obj.checked) {
                    Text(obj.source == "tap" ? "\(obj.name) (added)" : "\(obj.name) #\(obj.id)")
                }
                Picker("Interaction", selection: $obj.mode) {
                    ForEach(CaptureMode.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.segmented)
                if obj.frames > 0 {
                    Text("seen in \(obj.frames) frames").font(.caption2).foregroundStyle(.secondary)
                }
            }
        }
    }
}

/// Tap the object on one of the room-pass photos, then name it.
struct AddObjectView: View {
    @ObservedObject var scan: ScanController
    @Environment(\.dismiss) private var dismiss
    @State private var picked: (frame: ScanController.Keyframe, point: CGPoint)?
    @State private var name = ""
    @State private var mode: CaptureMode = .pickable

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            if scan.keyframes.isEmpty {
                Text("No photos from the room pass to tap on.").foregroundStyle(.secondary)
            } else {
                Text(picked == nil ? "Tap the object on a photo." : "Tapped. Name it below.")
                    .font(.callout)
                ScrollView(.horizontal) {
                    HStack(spacing: 8) {
                        ForEach(scan.keyframes) { kf in
                            GeometryReader { geo in
                                Image(uiImage: kf.image).resizable().scaledToFit()
                                    .frame(width: geo.size.width, height: geo.size.height)
                                    .overlay(alignment: .topLeading) {
                                        if let p = picked, p.frame.id == kf.id {
                                            Circle().stroke(.yellow, lineWidth: 3).frame(width: 28, height: 28)
                                                .offset(x: p.point.x * geo.size.width - 14, y: p.point.y * geo.size.height - 14)
                                        }
                                    }
                                    .contentShape(Rectangle())
                                    .onTapGesture(coordinateSpace: .local) { loc in
                                        picked = (kf, CGPoint(x: loc.x / geo.size.width, y: loc.y / geo.size.height))
                                    }
                            }
                            .frame(width: 240, height: 320)
                        }
                    }
                }
            }
            TextField("What is it? e.g. cupboard, pressure cooker", text: $name)
                .textFieldStyle(.roundedBorder)
                .onChange(of: name) { mode = CaptureMode.forClass($0) }
            Picker("Interaction", selection: $mode) {
                ForEach(CaptureMode.allCases) { Text($0.label).tag($0) }
            }
            .pickerStyle(.segmented)
            Button("Add object") {
                if let p = picked {
                    scan.addTappedObject(keyframe: p.frame, at: p.point, name: name, mode: mode)
                    dismiss()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(picked == nil || name.trimmingCharacters(in: .whitespaces).isEmpty)
            Spacer()
        }
        .padding()
        .navigationTitle("Add object")
    }
}

/// Instructions and progress for the current guided capture.
struct GuideOverlay: View {
    @ObservedObject var scan: ScanController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(scan.guideTitle).font(.headline)
            Text(scan.guideText).font(.callout)
            if scan.guideMode == .pickable {
                HStack {
                    ZStack {
                        Circle().stroke(.white.opacity(0.3), lineWidth: 6)
                        Circle().trim(from: 0, to: CGFloat(scan.coverage))
                            .stroke(scan.coverage >= 0.8 ? .green : .yellow, style: StrokeStyle(lineWidth: 6, lineCap: .round))
                            .rotationEffect(.degrees(-90))
                    }
                    .frame(width: 44, height: 44)
                    Text(scan.coverage >= 0.8 ? "Covered \(Int(scan.coverage * 100)) % - press Done"
                         : "Walk around it: \(Int(scan.coverage * 100)) % of the way")
                        .font(.callout.monospacedDigit())
                }
            }
            if scan.guideMode == .transparent {
                let top = scan.guideStage == "top"
                let ok = top ? scan.pitch < -65 : abs(scan.pitch) < 15
                Label(String(format: "phone tilt %.0f°", scan.pitch) + (top ? " (point it straight down)" : " (hold it level)"),
                      systemImage: ok ? "checkmark.circle.fill" : "exclamationmark.triangle")
                    .foregroundStyle(ok ? .green : .yellow)
            }
            HStack {
                Spacer()
                Button("Skip this object") { scan.skipObject() }.font(.callout)
            }
        }
        .padding(12)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
        .padding(.horizontal)
    }
}
