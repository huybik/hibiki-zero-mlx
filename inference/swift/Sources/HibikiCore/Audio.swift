import AVFoundation
import Foundation

public enum Audio {
    public static let sampleRate = 24_000
    /// Samples per Mimi frame (80 ms at 24 kHz).
    public static let frameSize = 1920

    /// Reads the first channel of any file AVFoundation decodes, resampled to 24 kHz.
    public static func read(_ url: URL) throws -> [Float] {
        let file = try AVAudioFile(forReading: url)
        let source = file.processingFormat
        guard
            let input = AVAudioPCMBuffer(
                pcmFormat: source, frameCapacity: AVAudioFrameCount(file.length))
        else { throw HibikiError.unsupported("audio buffer for \(url.path)") }
        try file.read(into: input)
        let mono = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: source.sampleRate, channels: 1,
            interleaved: false)!
        let first = AVAudioPCMBuffer(pcmFormat: mono, frameCapacity: input.frameLength)!
        first.frameLength = input.frameLength
        first.floatChannelData![0].update(
            from: input.floatChannelData![0], count: Int(input.frameLength))
        if source.sampleRate == Double(sampleRate) {
            return Array(UnsafeBufferPointer(start: first.floatChannelData![0], count: Int(first.frameLength)))
        }
        let target = AVAudioFormat(
            commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1,
            interleaved: false)!
        guard let converter = AVAudioConverter(from: mono, to: target) else {
            throw HibikiError.unsupported("resampling \(source.sampleRate) Hz")
        }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        let capacity = AVAudioFrameCount(
            (Double(first.frameLength) * Double(sampleRate) / source.sampleRate).rounded(.up) + 1024)
        let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity)!
        var consumed = false
        var error: NSError?
        let status = converter.convert(to: output, error: &error) { _, inputStatus in
            if consumed {
                inputStatus.pointee = .endOfStream
                return nil
            }
            consumed = true
            inputStatus.pointee = .haveData
            return first
        }
        if status == .error { throw error ?? HibikiError.unsupported("resampling") }
        return Array(UnsafeBufferPointer(start: output.floatChannelData![0], count: Int(output.frameLength)))
    }

    /// Writes 24 kHz mono float samples as a WAV file.
    public static func write(_ samples: [Float], to url: URL) throws {
        try Writer(url).append(samples)
    }

    /// A 24 kHz mono WAV file written as samples arrive.
    public final class Writer {
        private let file: AVAudioFile
        private let format = AVAudioFormat(
            standardFormatWithSampleRate: Double(Audio.sampleRate), channels: 1)!

        public init(_ url: URL) throws {
            file = try AVAudioFile(forWriting: url, settings: format.settings)
        }

        public func append(_ samples: [Float]) throws {
            guard !samples.isEmpty else { return }
            let buffer = AVAudioPCMBuffer(
                pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count))!
            buffer.frameLength = AVAudioFrameCount(samples.count)
            buffer.floatChannelData![0].update(from: samples, count: samples.count)
            try file.write(from: buffer)
        }
    }
}
