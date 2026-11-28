// hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N] [--lm-units U]
//   Translates an audio file, as `python inference/main.py INPUT` and training's evaluation do:
//   the input in whole frames (the last zero-padded), the source EOS (`Translator.endInput`),
//   then silence for 20 s in all, stopping at the first frame after `endInput` where the model
//   emits the text EOS. An EOS before that, which the model wrote before hearing the source EOS,
//   stops nothing, so a long file is translated to its end. Its speech ends max_delay - 1 frames
//   before main.py's, which steps on past the EOS for the speech of the frames before it.
//
// hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE] [--lm-units U]
//   Loops INPUT for M minutes (default 10) through `Bench`, paced like a live microphone unless
//   --unpaced, writes BASE.{jsonl,wav,txt} and prints a per-minute summary.
//
// --model is a directory export_lm_coreml.py wrote (default
// weights/student_slices14/pal6_snap6_ane6f, the slices student Speech Live ships).
// --lm-units ane|gpu|cpu|all: where the Core ML LM runs (default ane, the Neural Engine).

import Foundation
import HibikiCore

let usage = """
    usage: hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N]
                            [--lm-units ane|gpu|cpu|all]
           hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out BASE]
                        [--lm-units ane|gpu|cpu|all]

    """
// Frames after the input, the source EOS frame included: training's evaluation generates its
// longest clip plus 20 s (file_mode.py TAIL_S).
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
guard let lmUnits = Translator.lmComputeUnitNames[option("--lm-units") ?? "ane"] else { fail() }

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
    /// Keeps a frame's output; true if the model emitted the text EOS in it.
    @discardableResult
    func take(_ out: Translator.Output) throws -> Bool {
        frames += 1
        timing += out.timing
        text += out.text
        print(out.text, terminator: "")
        fflush(stdout)
        try wav.append(out.pcm)
        ended = out.textToken == translator.vocabulary.eos
        return ended
    }
    run: do {
        for idx in 0..<steps {
            try take(translator.step(Array(source[(idx * fs)..<((idx + 1) * fs)])))
        }
        // The frame that takes the source EOS hears it only from the next frame on.
        try take(translator.endInput())
        for _ in 1..<tailFrames {
            if try take(translator.step([Float](repeating: 0, count: fs))) { break run }
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
        + "encode \(ms(timing.encode)), encode wait \(ms(timing.encodeWait)), "
        + "LM wait \(ms(timing.lmWait)); LM: trunk \(ms(timing.lmTrunk)), "
        + "audio head \(ms(timing.lmHeads)), host \(ms(timing.lmHost))"
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
        translator, clip: pcm, minutes: minutes, paced: !args.contains("--unpaced"), output: base
    ) { sample in
        if sample.ends(every: 750) {
            print(
                String(
                    format: "minute %d: %.1f ms/frame, lag %.2f s",
                    (sample.frame + 1) / 750, sample.totalMs, sample.lag))
        }
    }
    print(Bench.summary(samples))
    print("results: \(base.path).{jsonl,wav,txt}")
}

do {
    // Translator first: it configures MLX before MLX creates its Metal device, which seeding
    // would do.
    let translator = try Translator(
        directory: URL(
            fileURLWithPath: option("--model") ?? "weights/student_slices14/pal6_snap6_ane6f"),
        lmComputeUnits: lmUnits)
    translator.seed(seed)
    let pcm = try Audio.read(input)
    try args[0] == "translate" ? translate(translator, pcm) : bench(translator, pcm)
} catch {
    FileHandle.standardError.write("error: \(error)\n".data(using: .utf8)!)
    exit(1)
}
