import AppKit
import AVFoundation
import SwiftUI
import UniformTypeIdentifiers

final class InferenceController: ObservableObject {
    private enum RunMode {
        case file
        case microphone
    }

    /// The model last chosen, kept across launches (a choice is saved only when made, so a model
    /// on an unmounted drive is found again once it is back).
    private static let modelDefaultsKey = "modelPath"

    @Published var modelURL: URL?
    /// The language spoken, nil to let the model guess; main.py's --speaking primes the direction.
    @Published var speaking: String?
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
    private var stopRequested = false
    private var textOutputURL: URL?
    private var pendingOutputAudioURL: URL?
    private var microphoneHeader = ""
    private var microphoneReady = false
    private var audioPlayer: AVAudioPlayer?

    var canTranslateFile: Bool {
        !isRunning && modelURL != nil && audioURL != nil && repositoryRoot != nil
    }

    var isFileRunning: Bool {
        isRunning && !isMicrophoneRunning
    }

    init() {
        repositoryRoot = Self.findRepositoryRoot()
        // Quitting the app ends its backend, which would otherwise keep the GPU and microphone.
        // The controller lives as long as the app, so the observer is never removed.
        _ = NotificationCenter.default.addObserver(
            forName: NSApplication.willTerminateNotification, object: nil, queue: nil
        ) { [weak self] _ in
            self?.process?.terminate()
        }
        guard let repositoryRoot else {
            status = "Repository not found"
            return
        }
        // The model last chosen, else the shipping two-way bundle where the README downloads it.
        let candidates = [
            UserDefaults.standard.string(forKey: Self.modelDefaultsKey).map {
                URL(fileURLWithPath: $0, isDirectory: true)
            },
            repositoryRoot.appendingPathComponent(
                "weights/release/twoway_grpo1400_mlx_q6", isDirectory: true
            ),
        ]
        modelURL = candidates.compactMap { $0 }.first {
            FileManager.default.fileExists(atPath: $0.appendingPathComponent("config.json").path)
        }
    }

    func chooseModel() {
        let panel = NSOpenPanel()
        panel.title = "Choose an MLX model directory"
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url {
            modelURL = url
            UserDefaults.standard.set(url.path, forKey: Self.modelDefaultsKey)
        }
    }

    func chooseAudio() {
        let panel = NSOpenPanel()
        panel.title = "Choose Vietnamese or English speech"
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
                repositoryRoot.appendingPathComponent("inference/main.py").path,
                audioURL.path,
                "--model", modelURL.path,
                "--out", output.path,
                "--text-out", textOutput.path,
            ] + speakingArguments,
            mode: .file,
            outputAudio: output,
            textOutput: textOutput
        )
    }

    func startMicrophone() {
        guard let repositoryRoot, let modelURL else { return }
        launch(
            arguments: [
                repositoryRoot.appendingPathComponent("inference/main.py").path,
                "--mic",
                "--model", modelURL.path,
            ] + speakingArguments,
            mode: .microphone
        )
    }

    private var speakingArguments: [String] {
        speaking.map { ["--speaking", $0] } ?? []
    }

    func stop() {
        guard let process, process.isRunning else { return }
        status = "Stopping…"
        stopRequested = true
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
        status = mode == .file ? "Translating…" : "Starting microphone…"
        isRunning = true
        isMicrophoneRunning = mode == .microphone
        runMode = mode
        stopRequested = false
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
            guard !data.isEmpty else { return }
            DispatchQueue.main.async {
                self?.consume(data)
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

    private func consume(_ data: Data) {
        // Python writes each streamed character whole in one small write, which a pipe never
        // splits, so only a long write (a traceback) can end mid-character, shown as U+FFFD.
        let text = String(decoding: data, as: UTF8.self)
        guard !text.isEmpty else { return }
        appendLog(text)
        guard runMode == .microphone else { return }
        if microphoneReady {
            transcript.append(text)
            return
        }
        microphoneHeader.append(text)
        // The transcript starts after the blank line that ends main.py's "listening" line; a
        // traceback printed while the model loads has blank lines of its own.
        guard let listening = microphoneHeader.range(of: "listening ") else { return }
        status = "Listening"
        guard let separator = microphoneHeader.range(
            of: "\n\n", range: listening.upperBound..<microphoneHeader.endIndex
        ) else { return }
        microphoneReady = true
        transcript.append(String(microphoneHeader[separator.upperBound...]))
        microphoneHeader = ""
    }

    private func finish(exitCode: Int32) {
        // The process has exited, so the pipe ends: read what the handler had not yet.
        if let handle = pipe?.fileHandleForReading {
            handle.readabilityHandler = nil
            consume(handle.readDataToEndOfFile())
        }
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
        } else if completedMode == .microphone, exitCode == 0 || stopRequested {
            // Stopped while the model loads, Python ends on the signal instead of exiting 0.
            transcript = transcript.replacingOccurrences(of: "\n[stopping]", with: "")
            status = "Microphone stopped"
        } else if stopRequested {
            status = "Stopped"
        } else {
            status = "Inference failed (exit \(exitCode))"
        }
        self.textOutputURL = nil
        pendingOutputAudioURL = nil
    }

    private func appendLog(_ text: String) {
        log.append(text)
        if log.count > 40_000 {
            log.removeFirst(log.count - 40_000)
        }
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
}
