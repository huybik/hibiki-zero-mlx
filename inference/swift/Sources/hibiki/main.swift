// hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N] [--chunk K]
//   Translates an audio file, as `python inference/main.py INPUT`: the input is followed by up
//   to 8 s of silence to flush the lag, and generation stops once the input is consumed and the
//   model emits text EOS or 12 frames of special tokens (at the end of that step).
//
// hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE] [--chunk K]
//   Loops INPUT for M minutes (default 10) through `Bench`, paced like a live microphone unless
//   --unpaced, writes BASE.{jsonl,wav,txt} and prints a per-minute summary.
//
// --chunk K feeds K frames per `Translator.step` (default 1): Mimi then encodes and decodes K
// frames per call, for less CPU time per frame and more latency. The translation is the same,
// but `translate` stops only at the end of a step, up to K - 1 frames after K = 1 would.

import Foundation
import HibikiCore
import MLXRandom

let usage = """
    usage: hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N] [--chunk K]
           hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE] [--chunk K]

    """
let tailFrames = 100  // 8 s
let padStop = 12

let args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
func fail() -> Never {
    FileHandle.standardError.write(usage.data(using: .utf8)!)
    exit(2)
}
func number<T: LosslessStringConvertible>(_ name: String, default value: T) -> T {
    guard let text = option(name) else { return value }
    guard let parsed = T(text) else { fail() }
    return parsed
}
guard args.count >= 2, ["translate", "bench"].contains(args[0]) else { fail() }
let input = URL(fileURLWithPath: args[1])
let stem = input.deletingPathExtension().lastPathComponent
let seed = number("--seed", default: UInt64(299_792_458))
let minutes = number("--minutes", default: 10.0)
let chunk = number("--chunk", default: 1)
guard chunk > 0 else { fail() }

func translate(_ translator: Translator, _ pcm: [Float]) throws {
    let output = URL(fileURLWithPath: option("--out") ?? "translations/\(stem)_swift.wav")
    let steps = pcm.count / Audio.frameSize
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    let wav = try Audio.Writer(output)
    var text = ""
    var timing = Translator.Timing()
    var frames = 0
    var padRun = 0
    let start = Date()
    steps: for idx in stride(from: 0, to: steps + tailFrames, by: chunk) {
        let input = (idx..<min(idx + chunk, steps + tailFrames)).flatMap { frame in
            frame < steps
                ? Array(pcm[(frame * Audio.frameSize)..<((frame + 1) * Audio.frameSize)])
                : [Float](repeating: 0, count: Audio.frameSize)
        }
        let out = translator.step(input)
        frames += out.textTokens.count
        timing += out.timing
        text += out.text
        print(out.text, terminator: "")
        fflush(stdout)
        try wav.append(out.pcm)
        for (frame, token) in zip(idx..., out.textTokens) where frame >= steps {
            if token == translator.vocabulary.eos { break steps }
            padRun = translator.vocabulary.text(token) == nil ? padRun + 1 : 0
            if padRun >= padStop { break steps }
        }
    }
    try wav.append(translator.flush())
    let wall = Date().timeIntervalSince(start)
    print()
    let textURL = output.deletingPathExtension().appendingPathExtension("txt")
    try (text.trimmingCharacters(in: .whitespaces) + "\n").write(
        to: textURL, atomically: true, encoding: .utf8)
    let ms = { (s: Double) in String(format: "%.1f", 1000 * s / Double(frames)) }
    let speed = String(format: "%.2f s -> %.2fx RT", wall, Double(steps) / wall / 12.5)
    let stages = "LM start \(ms(timing.lmStart)), decode \(ms(timing.decode)), "
        + "encode \(ms(timing.encode)), LM wait \(ms(timing.lmWait))"
    print("[\(frames) frames (\(steps) audio) in \(speed); ms/frame: \(stages)]")
    print("out: \(output.path), text: \(textURL.path)")
}

func bench(_ translator: Translator, _ pcm: [Float]) throws {
    let base = URL(fileURLWithPath: option("--out") ?? "translations/\(stem)_bench")
    try FileManager.default.createDirectory(
        at: base.deletingLastPathComponent(), withIntermediateDirectories: true)
    let samples = try Bench.run(
        translator, clip: pcm, minutes: minutes, framesPerStep: chunk,
        paced: !args.contains("--unpaced"), output: base
    ) { sample in
        if sample.ends(every: 750) {
            print(
                String(
                    format: "minute %d: %.1f ms/frame, lag %.2f s",
                    (sample.frame + sample.frames) / 750, sample.msPerFrame, sample.lag))
        }
    }
    print(Bench.summary(samples))
    print("results: \(base.path).{jsonl,wav,txt}")
}

do {
    // Translator first: it configures MLX before MLX creates its Metal device, which seeding
    // would do.
    let translator = try Translator(
        directory: URL(fileURLWithPath: option("--model") ?? "weights"))
    MLXRandom.seed(seed)
    let pcm = try Audio.read(input)
    try args[0] == "translate" ? translate(translator, pcm) : bench(translator, pcm)
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
