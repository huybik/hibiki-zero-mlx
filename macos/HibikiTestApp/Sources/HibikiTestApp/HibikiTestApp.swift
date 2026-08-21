import AppKit
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

@main
struct HibikiTestApp: App {
    @StateObject private var controller = InferenceController()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(controller)
        }
        .defaultSize(width: 780, height: 680)
    }
}

struct ContentView: View {
    @EnvironmentObject private var controller: InferenceController

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            modelPicker
            fileControls
            microphoneControls
            transcript
            backendLog
        }
        .padding(22)
        .frame(minWidth: 720, minHeight: 620)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Hibiki Test")
                    .font(.largeTitle.bold())
                Text("Vietnamese speech → English speech and text")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(controller.status)
                .font(.callout.weight(.semibold))
                .foregroundStyle(controller.isRunning ? .orange : .secondary)
                .padding(.horizontal, 11)
                .padding(.vertical, 6)
                .background(.quaternary, in: Capsule())
        }
    }

    private var modelPicker: some View {
        GroupBox("MLX model") {
            HStack(spacing: 12) {
                Image(systemName: "cpu")
                    .foregroundStyle(.secondary)
                Text(controller.modelURL?.path ?? "Choose a staged q4 or bf16 model directory")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                Button("Choose…", action: controller.chooseModel)
                    .disabled(controller.isRunning)
            }
            .padding(.vertical, 4)
        }
    }

    private var fileControls: some View {
        GroupBox("Translate an audio file") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Image(systemName: "waveform")
                        .foregroundStyle(.secondary)
                    Text(controller.audioURL?.path ?? "Choose a WAV, MP3, or FLAC file")
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Button("Choose audio…", action: controller.chooseAudio)
                        .disabled(controller.isRunning)
                }
                HStack {
                    Button(action: controller.translateFile) {
                        Label("Translate file", systemImage: "character.bubble")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!controller.canTranslateFile)

                    Button(action: controller.playOutput) {
                        Label("Play output", systemImage: "play.fill")
                    }
                    .disabled(controller.outputAudioURL == nil || controller.isRunning)

                    Button(action: controller.revealOutput) {
                        Label("Show in Finder", systemImage: "folder")
                    }
                    .disabled(controller.outputAudioURL == nil)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var microphoneControls: some View {
        GroupBox("Live microphone") {
            HStack {
                if controller.isMicrophoneRunning {
                    Button(action: controller.stop) {
                        Label("Stop microphone", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Button(action: controller.startMicrophone) {
                        Label("Start microphone", systemImage: "mic.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(controller.isRunning || controller.modelURL == nil)
                }
                Text("Uses the current macOS input and output devices.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.vertical, 4)
        }
    }

    private var transcript: some View {
        GroupBox("English transcript") {
            ScrollView {
                Text(controller.transcript.isEmpty ? "Translation will appear here." : controller.transcript)
                    .foregroundStyle(controller.transcript.isEmpty ? .secondary : .primary)
                    .font(.system(.body, design: .rounded))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(8)
            }
            .frame(minHeight: 120)
        }
    }

    private var backendLog: some View {
        DisclosureGroup("Backend log") {
            ScrollView {
                Text(controller.log.isEmpty ? "No backend output yet." : controller.log)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(height: 72)
            .padding(.top, 6)
        }
    }
}

final class InferenceController: ObservableObject {
    private enum RunMode {
        case file
        case microphone
    }

    @Published var modelURL: URL?
    @Published var audioURL: URL?
    @Published var outputAudioURL: URL?
    @Published var transcript = ""
    @Published var log = ""
    @Published var status = "Ready"
    @Published var isRunning = false
    @Published var isMicrophoneRunning = false

    private let pythonURL = URL(
        fileURLWithPath: "/opt/homebrew/Caskroom/miniconda/base/bin/python"
    )
    private let repositoryRoot: URL?
    private var process: Process?
    private var pipe: Pipe?
    private var runMode: RunMode?
    private var textOutputURL: URL?
    private var pendingOutputAudioURL: URL?
    private var microphoneHeader = ""
    private var microphoneReady = false
    private var audioPlayer: AVAudioPlayer?

    var canTranslateFile: Bool {
        !isRunning && modelURL != nil && audioURL != nil && repositoryRoot != nil
    }

    init() {
        repositoryRoot = Self.findRepositoryRoot()
        guard let repositoryRoot else {
            status = "Repository not found"
            return
        }
        let preferredModel = repositoryRoot
            .appendingPathComponent("weights/vi-step135000-mlx-bf16")
        if FileManager.default.fileExists(atPath: preferredModel.path) {
            modelURL = preferredModel
        }
        let sample = repositoryRoot.appendingPathComponent(
            "remote_dataset/fleurs_vi_en/validation/vi/vi_01624.wav"
        )
        if FileManager.default.fileExists(atPath: sample.path) {
            audioURL = sample
        }
    }

    deinit {
        if process?.isRunning == true {
            process?.terminate()
        }
    }

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.title = "Choose an MLX model directory"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK {
            modelURL = panel.url
        }
    }

    func chooseAudio() {
        let panel = NSOpenPanel()
        panel.title = "Choose Vietnamese audio"
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.allowedContentTypes = [.audio]
        if panel.runModal() == .OK {
            audioURL = panel.url
        }
    }

    func translateFile() {
        guard let repositoryRoot, let modelURL, let audioURL else { return }
        let translations = repositoryRoot.appendingPathComponent("translations", isDirectory: true)
        do {
            try FileManager.default.createDirectory(
                at: translations,
                withIntermediateDirectories: true
            )
        } catch {
            fail("Could not create translations directory: \(error.localizedDescription)")
            return
        }
        let stem = audioURL.deletingPathExtension().lastPathComponent
        let suffix = Int(Date().timeIntervalSince1970)
        let output = translations.appendingPathComponent("gui_\(stem)_\(suffix).wav")
        let textOutput = output.deletingPathExtension().appendingPathExtension("txt")
        outputAudioURL = nil
        launch(
            arguments: [
                repositoryRoot.appendingPathComponent("main.py").path,
                audioURL.path,
                "--model", modelURL.path,
                "--out", output.path,
                "--text-out", textOutput.path,
            ],
            mode: .file,
            outputAudio: output,
            textOutput: textOutput
        )
    }

    func startMicrophone() {
        guard let repositoryRoot, let modelURL else { return }
        launch(
            arguments: [
                repositoryRoot.appendingPathComponent("main.py").path,
                "--mic",
                "--model", modelURL.path,
            ],
            mode: .microphone
        )
    }

    func stop() {
        guard let process, process.isRunning else { return }
        status = "Stopping…"
        process.interrupt()
    }

    func playOutput() {
        guard let outputAudioURL else { return }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: outputAudioURL)
            audioPlayer?.prepareToPlay()
            audioPlayer?.play()
            status = "Playing output"
        } catch {
            fail("Could not play output: \(error.localizedDescription)")
        }
    }

    func revealOutput() {
        guard let outputAudioURL else { return }
        NSWorkspace.shared.activateFileViewerSelecting([outputAudioURL])
    }

    private func launch(
        arguments: [String],
        mode: RunMode,
        outputAudio: URL? = nil,
        textOutput: URL? = nil
    ) {
        guard process == nil, let repositoryRoot else { return }
        guard FileManager.default.isExecutableFile(atPath: pythonURL.path) else {
            fail("Python was not found at \(pythonURL.path)")
            return
        }

        transcript = ""
        log = ""
        status = mode == .file ? "Loading model…" : "Starting microphone…"
        isRunning = true
        isMicrophoneRunning = mode == .microphone
        runMode = mode
        pendingOutputAudioURL = outputAudio
        textOutputURL = textOutput
        microphoneHeader = ""
        microphoneReady = false

        let pipe = Pipe()
        let process = Process()
        process.executableURL = pythonURL
        process.arguments = arguments
        process.currentDirectoryURL = repositoryRoot
        process.standardOutput = pipe
        process.standardError = pipe
        var environment = ProcessInfo.processInfo.environment
        environment["PYTHONUNBUFFERED"] = "1"
        process.environment = environment

        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty, let text = String(data: data, encoding: .utf8) else { return }
            DispatchQueue.main.async {
                self?.consume(text)
            }
        }
        process.terminationHandler = { [weak self] completed in
            DispatchQueue.main.async {
                self?.finish(exitCode: completed.terminationStatus)
            }
        }

        do {
            try process.run()
            self.pipe = pipe
            self.process = process
        } catch {
            pipe.fileHandleForReading.readabilityHandler = nil
            fail("Could not start inference: \(error.localizedDescription)")
        }
    }

    private func consume(_ text: String) {
        log.append(text)
        if log.count > 40_000 {
            log.removeFirst(log.count - 40_000)
        }
        guard runMode == .microphone else { return }
        if microphoneReady {
            transcript.append(text)
            return
        }
        microphoneHeader.append(text)
        if microphoneHeader.contains("listening ") {
            status = "Listening"
        }
        guard let separator = microphoneHeader.range(of: "\n\n") else { return }
        microphoneReady = true
        transcript.append(String(microphoneHeader[separator.upperBound...]))
        microphoneHeader = ""
    }

    private func finish(exitCode: Int32) {
        pipe?.fileHandleForReading.readabilityHandler = nil
        let completedMode = runMode
        process = nil
        pipe = nil
        runMode = nil
        isRunning = false
        isMicrophoneRunning = false

        if completedMode == .file, exitCode == 0, let textOutputURL {
            do {
                transcript = try String(contentsOf: textOutputURL, encoding: .utf8)
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                outputAudioURL = pendingOutputAudioURL
                status = "Translation complete"
            } catch {
                fail("Translation finished but its transcript could not be read")
            }
        } else if completedMode == .microphone, exitCode == 0 {
            transcript = transcript.replacingOccurrences(of: "\n[stopping]", with: "")
            status = "Microphone stopped"
        } else {
            status = "Inference failed (exit \(exitCode))"
        }
        self.textOutputURL = nil
        pendingOutputAudioURL = nil
    }

    private func fail(_ message: String) {
        status = message
        isRunning = false
        isMicrophoneRunning = false
        process = nil
        pipe = nil
        runMode = nil
        pendingOutputAudioURL = nil
    }

    private static func findRepositoryRoot() -> URL? {
        let fileManager = FileManager.default
        let starts = [
            URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true),
            Bundle.main.bundleURL,
        ]
        for start in starts {
            var candidate = start
            for _ in 0..<8 {
                let main = candidate.appendingPathComponent("main.py")
                let runtime = candidate.appendingPathComponent("hibiki_mlx", isDirectory: true)
                var isDirectory: ObjCBool = false
                if fileManager.fileExists(atPath: main.path),
                   fileManager.fileExists(atPath: runtime.path, isDirectory: &isDirectory),
                   isDirectory.boolValue {
                    return candidate
                }
                candidate.deleteLastPathComponent()
            }
        }
        return nil
    }
}
