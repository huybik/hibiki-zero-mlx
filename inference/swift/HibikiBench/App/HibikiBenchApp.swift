// HibikiBench: the on-device sustained-load test of the Swift runtime.
//
// Reads Documents/model/ (Documents/NAME/ with `--model NAME`: config.json, weights, Mimi,
// tokenizer) and Documents/clip.wav, runs `Bench` for `--minutes M` (default 10), `--chunk K`
// frames per step (default 1), paced like a live microphone unless `--unpaced`, and writes
// Documents/bench-<time>.{jsonl,wav,txt} as it goes. Progress and the summary also go to stdout, which `xcrun devicectl device process launch
// --console` shows. Keep the app in the foreground: iOS refuses GPU work from the background, so
// locking the phone ends the run. A model with the direction condition translates in
// `--direction D` (vi-en or en-vi); without the flag, a picker asks for it before the run starts.

import HibikiCore
import SwiftUI

@main
struct HibikiBenchApp: App {
    @StateObject private var run = BenchRun()

    var body: some Scene {
        WindowGroup {
            VStack(alignment: .leading, spacing: 12) {
                if !run.directions.isEmpty && !run.started {
                    directionPicker
                }
                ScrollView {
                    Text(run.lines.joined(separator: "\n"))
                        .font(.system(.footnote, design: .monospaced))
                        .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
            .padding()
            .onAppear { run.prepare() }
        }
    }

    /// A model with the direction condition is told which way to translate before the run.
    private var directionPicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Direction", selection: $run.direction) {
                ForEach(run.directions, id: \.self) { direction in
                    Text(Self.label(direction)).tag(Optional(direction))
                }
            }
            .pickerStyle(.segmented)
            Button("Start", action: run.start)
                .buttonStyle(.borderedProminent)
                .disabled(run.direction == nil)
        }
    }

    /// "vi-en" as "Vietnamese → English".
    private static func label(_ direction: String) -> String {
        direction.split(separator: "-").map {
            Locale.current.localizedString(forLanguageCode: String($0)) ?? String($0)
        }.joined(separator: " → ")
    }
}

@MainActor
final class BenchRun: ObservableObject {
    /// The newest lines only: a text view that grows by a line every 10 s grew the app's
    /// footprint by ~1.6 MB a minute.
    @Published private(set) var lines: [String] = []
    /// The model's directions (its config.json's direction table); empty for a model without the
    /// direction condition, which hears the direction.
    @Published private(set) var directions: [String] = []
    /// The direction the run translates in: `--direction D`, else the picker's.
    @Published var direction: String?
    @Published private(set) var started = false
    private var prepared = false
    private let args = ProcessInfo.processInfo.arguments
    private let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]

    private func option(_ name: String) -> String? {
        args.firstIndex(of: name).flatMap { $0 + 1 < args.count ? args[$0 + 1] : nil }
    }

    private var model: String { option("--model") ?? "model" }

    /// Reads the model's directions, then starts at once unless the model needs a direction
    /// and the launch gave none: then the picker waits for one.
    func prepare() {
        guard !prepared else { return }
        prepared = true
        do {
            directions = try HibikiConfig.load(directory: docs.appending(path: model)).directions
        } catch {
            show("error: \(error)")
            return
        }
        direction = option("--direction")
        if directions.isEmpty || direction != nil { start() }
    }

    func start() {
        guard !started else { return }
        started = true
        UIApplication.shared.isIdleTimerDisabled = true
        let minutes = option("--minutes").flatMap(Double.init) ?? 10
        let chunk = option("--chunk").flatMap(Int.init) ?? 1
        let paced = !args.contains("--unpaced")
        let (docs, model, direction) = (docs, model, direction)
        let thread = Thread { [weak self] in
            Self.bench(
                docs: docs, model: model, direction: direction, minutes: minutes, chunk: chunk,
                paced: paced
            ) { line in
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
        docs: URL, model: String, direction: String?, minutes: Double, chunk: Int, paced: Bool,
        say: @escaping (String) -> Void
    ) {
        do {
            say("loading \(docs.appending(path: model).path)" + (direction.map { ", \($0)" } ?? ""))
            // Freed MLX buffers count toward the app's memory limit; keep few of them.
            let translator = try Translator(
                directory: docs.appending(path: model), direction: direction,
                cacheLimit: 256 << 20)
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
