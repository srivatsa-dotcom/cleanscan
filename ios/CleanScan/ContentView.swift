import RealityKit
import SwiftUI

struct ContentView: View {
    @StateObject private var scan = ScanController()
    @State private var showLibrary = false

    var body: some View {
        ZStack {
            ARViewContainer(scan: scan).ignoresSafeArea()

            VStack(spacing: 12) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(scan.status).font(.callout.weight(.semibold))
                        if scan.isRecording {
                            Text("\(scan.frameCount) frames · \(String(format: "%.1f", scan.maskedPercent))% living beings masked")
                                .font(.caption.monospacedDigit())
                        }
                        if scan.rigEnabled {
                            RigStatusView(rig: scan.rig, host: scan.rigHost)
                        }
                        if !scan.hasSegmentationModel {
                            Text("Animal model not bundled - using built-in cat/dog detector")
                                .font(.caption2).foregroundStyle(.yellow)
                        }
                    }
                    .padding(10)
                    .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12))
                    Spacer()
                    if let preview = scan.maskPreview, scan.isRecording {
                        Image(uiImage: preview)
                            .resizable().scaledToFit()
                            .frame(width: 90)
                            .background(Color.black.opacity(0.4))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                            .overlay(alignment: .bottom) {
                                Text("removed").font(.caption2).padding(2)
                            }
                    }
                }
                .padding(.horizontal)

                Spacer()

                HStack(spacing: 28) {
                    Button { showLibrary = true } label: {
                        Image(systemName: "folder").font(.title2).frame(width: 56, height: 56)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                    .disabled(scan.isRecording)

                    Button(action: scan.toggleRecording) {
                        ZStack {
                            Circle().stroke(.white, lineWidth: 4).frame(width: 78, height: 78)
                            RoundedRectangle(cornerRadius: scan.isRecording ? 8 : 34)
                                .fill(.red)
                                .frame(width: scan.isRecording ? 34 : 66, height: scan.isRecording ? 34 : 66)
                        }
                    }

                    Button { scan.showMesh.toggle() } label: {
                        Image(systemName: scan.showMesh ? "cube.transparent.fill" : "cube.transparent")
                            .font(.title2).frame(width: 56, height: 56)
                            .background(.ultraThinMaterial, in: Circle())
                    }
                }
                .padding(.bottom, 24)
            }
            .foregroundStyle(.white)
        }
        .sheet(isPresented: $showLibrary) { LibraryView(scan: scan) }
    }
}

struct ARViewContainer: UIViewRepresentable {
    let scan: ScanController

    func makeUIView(context: Context) -> ARView {
        let view = ARView(frame: .zero, cameraMode: .ar, automaticallyConfigureSession: false)
        scan.attach(view)
        return view
    }

    func updateUIView(_ uiView: ARView, context: Context) {}
}

/// One line on the scan screen: is the LiDAR rig streaming?
struct RigStatusView: View {
    @ObservedObject var rig: RigClient
    let host: String

    var body: some View {
        if rig.connected {
            Text("Rig: \(rig.sensor.uppercased()) \(rig.kind) · \(rig.pointsPerSecond) pts/s"
                 + (rig.rttMs.map { String(format: " · sync %.0f ms", $0) } ?? ""))
                .font(.caption.monospacedDigit())
                .foregroundStyle(rig.pointsPerSecond > 0 ? .green : .yellow)
        } else {
            Text("Rig: looking for \(host) - join the CleanScanRig Wi-Fi")
                .font(.caption).foregroundStyle(.yellow)
        }
    }
}

struct LibraryView: View {
    @ObservedObject var scan: ScanController
    @State private var scans: [URL] = ScanLibrary.list()
    @State private var shareItem: ShareItem?
    @State private var zipping: URL?
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List {
                Section(header: Text("External LiDAR rig"),
                        footer: Text("Raspberry Pi + LD06/LD19 or TF-Luna on the phone (see rig/README.md). " +
                                     "Join the rig's Wi-Fi first; its address is 10.42.0.1 by default.")) {
                    Toggle("Use LiDAR rig", isOn: $scan.rigEnabled)
                    if scan.rigEnabled {
                        TextField("Rig address", text: $scan.rigHost)
                            .keyboardType(.numbersAndPunctuation)
                            .textInputAutocapitalization(.never)
                            .autocorrectionDisabled()
                        RigStatusView(rig: scan.rig, host: scan.rigHost)
                    }
                }
                Section(footer: Text("Export a scan and process it on your computer:\n" +
                                     "cleanscan process Scan-XXXX.zip -o my_world")) {
                    ForEach(scans, id: \.self) { url in
                        HStack {
                            VStack(alignment: .leading) {
                                Text(url.lastPathComponent).font(.body.monospaced())
                                Text("\(ScanLibrary.frameCount(url)) frames").font(.caption).foregroundStyle(.secondary)
                            }
                            Spacer()
                            if zipping == url {
                                ProgressView()
                            } else {
                                Button("Export") { export(url) }.buttonStyle(.bordered)
                            }
                        }
                    }
                    .onDelete { idx in
                        idx.map { scans[$0] }.forEach(ScanLibrary.delete)
                        scans = ScanLibrary.list()
                    }
                }
            }
            .navigationTitle("Scans")
            .overlay { if scans.isEmpty { Text("No scans yet").foregroundStyle(.secondary) } }
            .alert("Export failed", isPresented: .constant(error != nil)) {
                Button("OK") { error = nil }
            } message: { Text(error ?? "") }
            .sheet(item: $shareItem) { item in ShareSheet(items: [item.url]) }
        }
    }

    private func export(_ url: URL) {
        zipping = url
        DispatchQueue.global(qos: .userInitiated).async {
            let result = Result { try ScanLibrary.zip(url) }
            DispatchQueue.main.async {
                zipping = nil
                switch result {
                case .success(let zip): shareItem = ShareItem(url: zip)
                case .failure(let e): error = e.localizedDescription
                }
            }
        }
    }
}

struct ShareItem: Identifiable {
    let url: URL
    var id: String { url.absoluteString }
}

struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ vc: UIActivityViewController, context: Context) {}
}
