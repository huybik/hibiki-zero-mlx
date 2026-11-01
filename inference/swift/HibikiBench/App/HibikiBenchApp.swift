// HibikiBench: the on-device sustained-load test of the Swift runtime.
//
// Reads Documents/model/ (config.json, weights, Mimi, tokenizer) and Documents/clip.wav, runs
// `Bench` for `--minutes M` (default 10), paced like a live microphone unless `--unpaced`, and
// writes Documents/bench-<time>.{json,txt,wav}. Progress and the summary also go to stdout,
// which `xcrun devicectl device process launch --console` shows.

import HibikiCore
import SwiftUI

@main
struct HibikiBenchApp: App {
    @StateObject private var run = BenchRun()

    var body: some Scene {
        WindowGroup {
            ScrollView {
                Text(run.log)
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
    @Published private(set) var log = ""
    private var started = false

    func start() {
        guard !started else { return }
        started = true
        UIApplication.shared.isIdleTimerDisabled = true
        let args = ProcessInfo.processInfo.arguments
        let minutes = args.firstIndex(of: "--minutes").flatMap { Double(args[$0 + 1]) } ?? 10
        let paced = !args.contains("--unpaced")
        let thread = Thread { [weak self] in
            Self.bench(minutes: minutes, paced: paced) { line in
                print(line)
                fflush(stdout)
                Task { @MainActor in self?.log += line + "\n" }
            }
        }
        thread.stackSize = 8 << 20
        thread.start()
    }

    nonisolated private static func bench(
        minutes: Double, paced: Bool, say: @escaping (String) -> Void
    ) {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        do {
            say("loading \(docs.appending(path: "model").path)")
            let translator = try Translator(directory: docs.appending(path: "model"))
            let clip = try Audio.read(docs.appending(path: "clip.wav"))
            say(String(format: "loaded; footprint %.0f MB", Bench.footprintMB()))
            say("running \(minutes) min, \(paced ? "paced at 12.5 frames/s" : "unpaced")")
            let result = Bench.run(translator, clip: clip, minutes: minutes, paced: paced) {
                sample in
                if sample.frame % 125 == 124 {
                    say(
                        String(
                            format: "%5.0f s  %5.1f ms/frame  lag %.2f s  thermal %d  %.0f MB",
                            sample.wall, sample.totalMs, sample.lag, sample.thermal,
                            sample.footprintMB))
                }
            }
            let stamp = ISO8601DateFormatter().string(from: Date())
                .replacingOccurrences(of: ":", with: "-")
            let base = docs.appending(path: "bench-\(stamp)")
            try JSONEncoder().encode(result.samples).write(
                to: base.appendingPathExtension("json"))
            let summary = Bench.summary(result.samples)
            try (summary + "\n\n" + result.text + "\n").write(
                to: base.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
            try Audio.write(result.pcm, to: base.appendingPathExtension("wav"))
            say(summary)
            say("wrote \(base.lastPathComponent).{json,txt,wav}")
        } catch {
            say("error: \(error)")
        }
    }
}
