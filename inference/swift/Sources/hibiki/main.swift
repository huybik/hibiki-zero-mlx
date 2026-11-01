// hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N]
//   Translates an audio file, as `python inference/main.py INPUT`: the input is followed by up
//   to 8 s of silence to flush the lag, and generation stops once the input is consumed and the
//   model emits text EOS or 12 frames of special tokens.
//
// hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out OUT.json]
//   Loops INPUT for M minutes (default 10) through `Bench`, paced like a live microphone unless
//   --unpaced, and prints a per-minute summary.

import Foundation
import HibikiCore
import MLXRandom

let usage = """
    usage: hibiki translate INPUT [--model DIR] [--out OUT.wav] [--seed N]
           hibiki bench INPUT [--model DIR] [--minutes M] [--unpaced] [--out OUT.json]

    """
let tailFrames = 100  // 8 s
let padStop = 12

let args = Array(CommandLine.arguments.dropFirst())
func option(_ name: String) -> String? {
    guard let i = args.firstIndex(of: name), i + 1 < args.count else { return nil }
    return args[i + 1]
}
guard args.count >= 2, ["translate", "bench"].contains(args[0]) else {
    FileHandle.standardError.write(usage.data(using: .utf8)!)
    exit(2)
}
let input = URL(fileURLWithPath: args[1])
let stem = input.deletingPathExtension().lastPathComponent
MLXRandom.seed(UInt64(option("--seed") ?? "299792458")!)
let translator = try Translator(directory: URL(fileURLWithPath: option("--model") ?? "weights"))
let pcm = try Audio.read(input)

func translate() throws {
    let output = URL(fileURLWithPath: option("--out") ?? "translations/\(stem)_swift.wav")
    let steps = pcm.count / Audio.frameSize
    var text = ""
    var out: [Float] = []
    var timing = Translator.Timing()
    var frames = 0
    var padRun = 0
    let start = Date()
    for idx in 0..<(steps + tailFrames) {
        let chunk =
            idx < steps
            ? Array(pcm[(idx * Audio.frameSize)..<((idx + 1) * Audio.frameSize)])
            : [Float](repeating: 0, count: Audio.frameSize)
        let frame = translator.step(chunk)
        frames += 1
        timing.encode += frame.timing.encode
        timing.main += frame.timing.main
        timing.depformer += frame.timing.depformer
        timing.decode += frame.timing.decode
        if let piece = frame.text {
            text += piece
            print(piece, terminator: "")
            fflush(stdout)
        }
        out += frame.pcm ?? []
        if idx >= steps {
            if frame.textToken == translator.vocabulary.eos { break }
            padRun = frame.text == nil ? padRun + 1 : 0
            if padRun >= padStop { break }
        }
    }
    let wall = Date().timeIntervalSince(start)
    print()
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    try Audio.write(out, to: output)
    let textURL = output.deletingPathExtension().appendingPathExtension("txt")
    try (text.trimmingCharacters(in: .whitespaces) + "\n").write(
        to: textURL, atomically: true, encoding: .utf8)
    let ms = { (s: Double) in String(format: "%.1f", 1000 * s / Double(frames)) }
    let speed = String(format: "%.2f s -> %.2fx RT", wall, Double(steps) / wall / 12.5)
    let stages = "encode \(ms(timing.encode)), main \(ms(timing.main)), "
        + "depformer \(ms(timing.depformer)), decode \(ms(timing.decode))"
    print("[\(frames) frames (\(steps) audio) in \(speed); ms/frame: \(stages)]")
    print("out: \(output.path), text: \(textURL.path)")
}

func bench() throws {
    let output = URL(fileURLWithPath: option("--out") ?? "translations/\(stem)_bench.json")
    let result = Bench.run(
        translator, clip: pcm, minutes: Double(option("--minutes") ?? "10")!,
        paced: !args.contains("--unpaced")
    ) { sample in
        if sample.frame % 750 == 749 {
            print(
                String(
                    format: "minute %d: %.1f ms/frame, lag %.2f s", (sample.frame + 1) / 750,
                    sample.totalMs, sample.lag))
        }
    }
    try FileManager.default.createDirectory(
        at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
    try JSONEncoder().encode(result.samples).write(to: output)
    print(Bench.summary(result.samples))
    print("text: \(result.text.prefix(400))…\nsamples: \(output.path)")
}

try args[0] == "translate" ? translate() : bench()
