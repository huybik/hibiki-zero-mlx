// HibikiBench: the on-device sustained-load test of the Swift runtime.
//
// Reads Documents/model/ (Documents/NAME/ with `--model NAME`: config.json, weights, Mimi,
// tokenizer) and Documents/clip.wav, runs `Bench` for `--minutes M` (default 10), `--chunk K`
// frames per step (default 1), paced like a live microphone unless `--unpaced`, and writes
// Documents/bench-<time>.{jsonl,wav,txt} as it goes. Progress and the summary also go to stdout, which `xcrun devicectl device process launch
// --console` shows. Keep the app in the foreground: iOS refuses GPU work from the background, so
// locking the phone ends the run.

import HibikiCore
import SwiftUI

@main
struct HibikiBenchApp: App {
    @StateObject private var run = BenchRun()

    var body: some Scene {
        WindowGroup {
            ScrollView {
                Text(run.lines.joined(separator: "\n"))
                    .font(.system(.footnote, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding()
            }
            .onAppear { run.start() }
        }
    }
}

@MainActor
final class BenchRun: ObservableObject {
    /// The newest lines only: a text view that grows by a line every 10 s grew the app's
    /// footprint by ~1.6 MB a minute.
    @Published private(set) var lines: [String] = []
    private var started = false

    func start() {
        guard !started else { return }
        started = true
        UIApplication.shared.isIdleTimerDisabled = true
        let args = ProcessInfo.processInfo.arguments
        let minutes = args.firstIndex(of: "--minutes")
            .flatMap { $0 + 1 < args.count ? Double(args[$0 + 1]) : nil } ?? 10
        let chunk = args.firstIndex(of: "--chunk")
            .flatMap { $0 + 1 < args.count ? Int(args[$0 + 1]) : nil } ?? 1
        let paced = !args.contains("--unpaced")
        let model = args.firstIndex(of: "--model")
            .flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil } ?? "model"
        let thread = Thread { [weak self] in
            Self.bench(model: model, minutes: minutes, chunk: chunk, paced: paced) { line in
                print(line)
                fflush(stdout)
                Task { @MainActor in self?.show(line) }
            }
        }
        thread.stackSize = 8 << 20
        thread.qualityOfService = .userInteractive  // as a live audio path would run
        thread.start()
    }

    private func show(_ line: String) {
        lines.append(line)
        lines.removeFirst(max(0, lines.count - 10))
    }

    nonisolated private static func bench(
        model: String, minutes: Double, chunk: Int, paced: Bool,
        say: @escaping (String) -> Void
    ) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            say("loading \(docs.appending(path: model).path)")
            // Freed MLX buffers count toward the app's memory limit; keep few of them.
            let translator = try Translator(
                directory: docs.appending(path: model), cacheLimit: 256 << 20)
            let clip = try Audio.read(docs.appending(path: "clip.wav"))
            say(String(format: "loaded; footprint %.0f MB", Bench.footprintMB()))
            say(
                "running \(minutes) min, \(chunk) frames per step, "
                    + (paced ? "paced at 12.5 frames/s" : "unpaced"))
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            let base = docs.appending(path: "bench-\(stamp)")
            let samples = try Bench.run(
                translator, clip: clip, minutes: minutes, framesPerStep: chunk, paced: paced,
                output: base
            ) { sample in
                if sample.ends(every: 125) {
                    say(
                        String(
                            format: "%5.0f s  %5.1f ms/frame  lag %.2f s  thermal %d  %.0f MB"
                                + " (MLX %.0f)",
                            sample.wall, sample.msPerFrame, sample.lag, sample.thermal,
                            sample.footprintMB, sample.mlxActiveMB))
                }
            }
            say(Bench.summary(samples))
            say("wrote \(base.lastPathComponent).{jsonl,wav,txt}")
        } catch {
            say("error: \(error)")
        }
    }
}
