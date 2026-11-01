import Foundation
import os

/// Sustained-load test: feeds a looped clip for a fixed duration, either paced like a live
/// microphone (one frame every 80 ms) or back to back, and records every frame's stage times,
/// lag, thermal state and memory.
public enum Bench {
    public struct Sample: Codable {
        public let frame: Int
        /// Seconds since the run started, when this frame finished.
        public let wall: Double
        /// Paced: seconds between the frame's arrival and its result (stays under 0.08 while
        /// the device keeps up, grows without bound when it does not). Unpaced: 0.
        public let lag: Double
        public let encodeMs: Double
        public let mainMs: Double
        public let depformerMs: Double
        public let decodeMs: Double
        /// `ProcessInfo.ThermalState`: 0 nominal, 1 fair, 2 serious, 3 critical.
        public let thermal: Int
        public let footprintMB: Double
        /// Memory the process may still allocate before the system kills it (iOS only).
        public let availableMB: Double?

        public var totalMs: Double { encodeMs + mainMs + depformerMs + decodeMs }
    }

    public struct Result {
        public let samples: [Sample]
        public let text: String
        public let pcm: [Float]
    }

    public static func run(
        _ translator: Translator, clip: [Float], minutes: Double, paced: Bool,
        progress: (Sample) -> Void = { _ in }
    ) -> Result {
        let frames = Int(minutes * 60 * 12.5)
        let clipFrames = clip.count / Audio.frameSize
        precondition(clipFrames > 0, "the clip is shorter than one frame")
        translator.reset()
        var samples: [Sample] = []
        var text = ""
        var pcm: [Float] = []
        samples.reserveCapacity(frames)
        let start = DispatchTime.now()
        for idx in 0..<frames {
            let arrival = Double(idx) / 12.5
            if paced {
                let wait = arrival - seconds(start, DispatchTime.now())
                if wait > 0 { Thread.sleep(forTimeInterval: wait) }
            }
            let offset = (idx % clipFrames) * Audio.frameSize
            let frame = translator.step(Array(clip[offset..<(offset + Audio.frameSize)]))
            let wall = seconds(start, DispatchTime.now())
            text += frame.text ?? ""
            pcm += frame.pcm ?? []
            let sample = Sample(
                frame: idx, wall: wall, lag: paced ? wall - arrival : 0,
                encodeMs: 1000 * frame.timing.encode, mainMs: 1000 * frame.timing.main,
                depformerMs: 1000 * frame.timing.depformer, decodeMs: 1000 * frame.timing.decode,
                thermal: ProcessInfo.processInfo.thermalState.rawValue,
                footprintMB: footprintMB(), availableMB: availableMB())
            samples.append(sample)
            progress(sample)
        }
        return Result(samples: samples, text: text, pcm: pcm)
    }

    /// One line per minute: frame-time percentiles, worst lag, thermal state and memory.
    public static func summary(_ samples: [Sample]) -> String {
        let perMinute = 750
        var lines = ["minute  frames  p50 ms  p90 ms  p99 ms  depformer p50  max lag s  thermal  footprint MB"]
        for start in stride(from: 0, to: samples.count, by: perMinute) {
            let chunk = Array(samples[start..<min(start + perMinute, samples.count)])
            let total = chunk.map(\.totalMs).sorted()
            let dep = chunk.map(\.depformerMs).sorted()
            let q = { (xs: [Double], p: Double) in xs[min(xs.count - 1, Int(p * Double(xs.count)))] }
            lines.append(
                String(
                    format: "%6d  %6d  %6.1f  %6.1f  %6.1f  %13.1f  %9.2f  %7d  %12.0f",
                    start / perMinute + 1, chunk.count, q(total, 0.5), q(total, 0.9),
                    q(total, 0.99), q(dep, 0.5), chunk.map(\.lag).max() ?? 0,
                    chunk.map(\.thermal).max() ?? 0, chunk.map(\.footprintMB).max() ?? 0))
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
