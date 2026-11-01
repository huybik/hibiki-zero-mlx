import Foundation
import MLX
import os

/// Sustained-load test: feeds a looped clip for a fixed duration, either paced like a live
/// microphone (frame i is available at (i + 1) × 80 ms) or back to back, and records every
/// frame's stage times, lag, thermal state and memory. Results go to disk as they are made,
/// so a run the system kills still leaves its data: `<base>.jsonl` (one sample per frame),
/// `<base>.wav` (the translation) and, at the end, `<base>.txt` (summary and text).
public enum Bench {
    public struct Sample: Codable {
        public let frame: Int
        /// Seconds since the run started, when this frame finished.
        public let wall: Double
        /// Paced: seconds between the end of the frame's 80 ms of audio and its result (stays
        /// under 0.08 while the device keeps up, grows without bound when it does not).
        /// Unpaced: 0.
        public let lag: Double
        /// `Translator.Timing`, in milliseconds.
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
    }

    public static func run(
        _ translator: Translator, clip: [Float], minutes: Double, paced: Bool, output base: URL,
        progress: (Sample) -> Void = { _ in }
    ) throws -> [Sample] {
        let frames = Int(minutes * 60 * 12.5)
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
        samples.reserveCapacity(frames)
        let start = DispatchTime.now()
        for idx in 0..<frames {
            let arrival = Double(idx + 1) / 12.5
            if paced {
                let wait = arrival - seconds(start, DispatchTime.now())
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            let offset = (idx % clipFrames) * Audio.frameSize
            let frame = translator.step(Array(clip[offset..<(offset + Audio.frameSize)]))
            let wall = seconds(start, DispatchTime.now())
            text += frame.text ?? ""
            try wav.append(frame.pcm ?? [])
            let sample = Sample(
                frame: idx, wall: wall, lag: paced ? wall - arrival : 0,
                lmStartMs: 1000 * frame.timing.lmStart, decodeMs: 1000 * frame.timing.decode,
                encodeMs: 1000 * frame.timing.encode, lmWaitMs: 1000 * frame.timing.lmWait,
                thermal: ProcessInfo.processInfo.thermalState.rawValue,
                footprintMB: footprintMB(), mlxActiveMB: Double(Memory.activeMemory) / 1_048_576,
                mlxCacheMB: Double(Memory.cacheMemory) / 1_048_576, availableMB: availableMB())
            samples.append(sample)
            try log.write(contentsOf: encoder.encode(sample) + Data("\n".utf8))
            progress(sample)
        }
        try wav.append(translator.flush() ?? [])
        try (summary(samples) + "\n\n" + text + "\n").write(
            to: base.appendingPathExtension("txt"), atomically: true, encoding: .utf8)
        return samples
    }

    /// One line per minute: frame-time percentiles, worst lag, thermal state and memory.
    public static func summary(_ samples: [Sample]) -> String {
        let perMinute = 750
        var lines = [
            "minute  frames  p50 ms  p90 ms  p99 ms  LM wait p50  max lag s  thermal"
                + "  footprint MB  MLX active MB"
        ]
        for start in stride(from: 0, to: samples.count, by: perMinute) {
            let chunk = Array(samples[start..<min(start + perMinute, samples.count)])
            let total = chunk.map(\.totalMs).sorted()
            let wait = chunk.map(\.lmWaitMs).sorted()
            let q = { (xs: [Double], p: Double) in xs[min(xs.count - 1, Int(p * Double(xs.count)))] }
            lines.append(
                String(
                    format: "%6d  %6d  %6.1f  %6.1f  %6.1f  %11.1f  %9.2f  %7d  %12.0f  %13.0f",
                    start / perMinute + 1, chunk.count, q(total, 0.5), q(total, 0.9),
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
