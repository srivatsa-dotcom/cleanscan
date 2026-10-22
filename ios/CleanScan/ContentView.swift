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
                    .disabled(!scan.lidarAvailable)

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
        .sheet(isPresented: $showLibrary) { LibraryView() }
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

struct LibraryView: View {
    @State private var scans: [URL] = ScanLibrary.list()
    @State private var shareItem: ShareItem?
    @State private var zipping: URL?
    @State private var error: String?

    var body: some View {
        NavigationStack {
            List {
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
