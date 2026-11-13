// hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N] [--chunk K]
//   Translates an audio file, as `python inference/main.py INPUT` and training's evaluation do:
//   the input in whole frames (the last zero-padded), the source EOS (`Translator.endInput`),
//   then silence for 20 s in all, stopping at the end of the step where the model emits the
//   text EOS (with --chunk K > 1 the text keeps what that step wrote after the EOS).
//
// hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE] [--chunk K]
//   Loops INPUT for M minutes (default 10) through `Bench`, paced like a live microphone unless
//   --unpaced, writes BASE.{jsonl,wav,txt} and prints a per-minute summary.
//
// --chunk K feeds K frames per `Translator.step` (default 1): Mimi then encodes K frames per
// call, for less CPU time per frame and more latency (its decoder runs one frame per call). The translation is the same,
// but `translate` stops only at the end of a step, up to K - 1 frames after K = 1 would.
//
// --model defaults to weights/hibiki-zero-3b-q4, upstream's 3B staged by convert_mlx.py --q4.

import Foundation
import HibikiCore
import MLXRandom

let usage = """
    usage: hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N] [--chunk K]
           hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE] [--chunk K]

    """
// Frames after the input, the source EOS frame included: training's evaluation generates its
// longest clip plus 20 s (pipeline.py TAIL_S).
let tailFrames = 250

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
    let fs = Audio.frameSize
    let steps = (pcm.count + fs - 1) / fs
    let source = pcm + [Float](repeating: 0, count: steps * fs - pcm.count)
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    let wav = try Audio.Writer(output)
    var text = ""
    var timing = Translator.Timing()
    var frames = 0
    var ended = false
    let start = Date()
    /// Keeps a step's output; true once the model has emitted the text EOS.
    func take(_ out: Translator.Output) throws -> Bool {
        frames += out.textTokens.count
        timing += out.timing
        text += out.text
        print(out.text, terminator: "")
        fflush(stdout)
        try wav.append(out.pcm)
        ended = out.textTokens.contains(translator.vocabulary.eos)
        return ended
    }
    run: do {
        for idx in stride(from: 0, to: steps, by: chunk) {
            let end = min(idx + chunk, steps)
            if try take(translator.step(Array(source[(idx * fs)..<(end * fs)]))) { break run }
        }
        if try take(translator.endInput()) { break run }
        for idx in stride(from: 1, to: tailFrames, by: chunk) {
            let count = min(chunk, tailFrames - idx)
            if try take(translator.step([Float](repeating: 0, count: count * fs))) { break run }
        }
    }
    let tail = try translator.flush()
    try wav.append(tail.pcm)
    text += tail.text
    let wall = Date().timeIntervalSince(start)
    print(tail.text)
    let textURL = output.deletingPathExtension().appendingPathExtension("txt")
    try (text.trimmingCharacters(in: .whitespaces) + "\n").write(
        to: textURL, atomically: true, encoding: .utf8)
    let ms = { (s: Double) in String(format: "%.1f", 1000 * s / Double(frames)) }
    let seconds = Double(pcm.count) / Double(Audio.sampleRate)
    let speed = String(format: "%.2f s -> %.2fx RT", wall, seconds / wall)
    let stages = "LM start \(ms(timing.lmStart)), decode \(ms(timing.decode)), "
        + "encode \(ms(timing.encode)), LM wait \(ms(timing.lmWait))"
    print(
        "[\(frames) frames for \(String(format: "%.2f", seconds)) s of audio "
            + "(\(ended ? "ended" : "no EOS")) in \(speed); ms/frame: \(stages)]")
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
        directory: URL(fileURLWithPath: option("--model") ?? "weights/hibiki-zero-3b-q4"))
    MLXRandom.seed(seed)
    let pcm = try Audio.read(input)
    try args[0] == "translate" ? translate(translator, pcm) : bench(translator, pcm)
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
