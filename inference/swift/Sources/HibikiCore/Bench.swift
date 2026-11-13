import Foundation
import MLX
import os

/// Sustained-load test: feeds a looped clip for a fixed duration, k frames per step, either
/// paced like a live microphone (frame i is available at (i + 1) × 80 ms, so a step can start
/// once its last frame is) or back to back, and records every step's stage times, lag, thermal
/// state and memory. Results go to disk as they are made, so a run the system kills still
/// leaves its data: `<base>.jsonl` (one sample per step), `<base>.wav` (the translation) and,
/// at the end, `<base>.txt` (summary and text).
public enum Bench {
    public struct Sample: Codable {
        /// The step's first frame, and its number of frames.
        public let frame: Int
        public let frames: Int
        /// Seconds since the run started, when this step finished.
        public let wall: Double
        /// Paced: seconds between the end of the step's last frame of audio and its result
        /// (stays under 0.08 × frames while the device keeps up, grows without bound when it
        /// does not). Unpaced: 0.
        public let lag: Double
        /// `Translator.Timing` of the whole step, in milliseconds.
        public let lmStartMs: Double
        public let decodeMs: Double
        public let encodeMs: Double
        public let lmWaitMs: Double
        /// `ProcessInfo.ThermalState`: 0 nominal, 1 fair, 2 serious, 3 critical.
        public let thermal: Int
        public let footprintMB: Double
        /// MLX buffers in use, and freed ones MLX keeps for reuse; both count in the footprint.
        public let mlxActiveMB: Double
        public let mlxCacheMB: Double
        /// Memory the process may still allocate before the system kills it (iOS only).
        public let availableMB: Double?

        public var totalMs: Double { lmStartMs + decodeMs + encodeMs + lmWaitMs }
        /// The step's time per frame; real time needs it under 80 ms.
        public var msPerFrame: Double { totalMs / Double(frames) }
        /// Whether this step completes a multiple of `every` frames, for progress lines.
        public func ends(every: Int) -> Bool { (frame + frames) / every > frame / every }
    }

    public static func run(
        _ translator: Translator, clip: [Float], minutes: Double, framesPerStep: Int,
        paced: Bool, output base: URL, progress: (Sample) -> Void = { _ in }
    ) throws -> [Sample] {
        precondition(framesPerStep > 0, "a step takes at least one frame")
        let frames = Int(minutes * 60 * 12.5) / framesPerStep * framesPerStep
        let clipFrames = clip.count / Audio.frameSize
        precondition(clipFrames > 0, "the clip is shorter than one frame")
        translator.reset()
        let jsonl = base.appendingPathExtension("jsonl")
        FileManager.default.createFile(atPath: jsonl.path, contents: nil)
        let log = try FileHandle(forWritingTo: jsonl)
        defer { try? log.close() }
        let wav = try Audio.Writer(base.appendingPathExtension("wav"))
        let encoder = JSONEncoder()
        var samples: [Sample] = []
        var text = ""
        samples.reserveCapacity(frames / framesPerStep)
        let start = DispatchTime.now()
        for idx in stride(from: 0, to: frames, by: framesPerStep) {
            let arrival = Double(idx + framesPerStep) / 12.5
            if paced {
                let wait = arrival - seconds(start, DispatchTime.now())
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            let pcm = (idx..<(idx + framesPerStep)).flatMap { frame in
                let offset = (frame % clipFrames) * Audio.frameSize
                return clip[offset..<(offset + Audio.frameSize)]
            }
            let out = try translator.step(pcm)
            let wall = seconds(start, DispatchTime.now())
            text += out.text
            try wav.append(out.pcm)
            let sample = Sample(
                frame: idx, frames: framesPerStep, wall: wall, lag: paced ? wall - arrival : 0,
                lmStartMs: 1000 * out.timing.lmStart, decodeMs: 1000 * out.timing.decode,
                encodeMs: 1000 * out.timing.encode, lmWaitMs: 1000 * out.timing.lmWait,
                thermal: ProcessInfo.processInfo.thermalState.rawValue,
                footprintMB: footprintMB(), mlxActiveMB: Double(Memory.activeMemory) / 1_048_576,
                mlxCacheMB: Double(Memory.cacheMemory) / 1_048_576, availableMB: availableMB())
            samples.append(sample)
            try log.write(contentsOf: encoder.encode(sample) + Data("\n".utf8))
            progress(sample)
        }
        let tail = try translator.flush()
        try wav.append(tail.pcm)
        text += tail.text
        try (summary(samples) + "\n\n" + text + "\n").write(
            to: base.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
        return samples
    }

    /// One line per minute of audio: percentiles of the time per frame and of the LM wait per
    /// frame over its steps, worst lag, thermal state and memory. A step counts toward the
    /// minute of its first frame, so with k not dividing 750 the rows hold a few frames more or
    /// less.
    public static func summary(_ samples: [Sample]) -> String {
        var lines = [
            "minute  frames  p50 ms  p90 ms  p99 ms  LM wait p50  max lag s  thermal"
                + "  footprint MB  MLX active MB"
        ]
        let minutes = Dictionary(grouping: samples) { $0.frame / 750 }
        for minute in minutes.keys.sorted() {
            let chunk = minutes[minute]!
            let total = chunk.map(\.msPerFrame).sorted()
            let wait = chunk.map { $0.lmWaitMs / Double($0.frames) }.sorted()
            let q = { (xs: [Double], p: Double) in xs[min(xs.count - 1, Int(p * Double(xs.count)))] }
            lines.append(
                String(
                    format: "%6d  %6d  %6.1f  %6.1f  %6.1f  %11.1f  %9.2f  %7d  %12.0f  %13.0f",
                    minute + 1, chunk.map(\.frames).reduce(0, +), q(total, 0.5), q(total, 0.9),
                    q(total, 0.99), q(wait, 0.5), chunk.map(\.lag).max() ?? 0,
                    chunk.map(\.thermal).max() ?? 0, chunk.map(\.footprintMB).max() ?? 0,
                    chunk.map(\.mlxActiveMB).max() ?? 0))
        }
        return lines.joined(separator: "\n")
    }

    public static func footprintMB() -> Double {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(
            MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1_048_576 : -1
    }

    public static func availableMB() -> Double? {
        #if os(iOS)
            Double(os_proc_available_memory()) / 1_048_576
        #else
            nil
        #endif
    }
}
