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

    /// A 24 kHz mono 16-bit WAV file written as samples arrive (clipped to [-1, 1], as
    /// `sphn.write_wav`). The header's sizes are rewritten after every append, so the file
    /// stays valid if the process dies mid-run.
    public final class Writer {
        private let handle: FileHandle
        private var dataBytes: UInt32 = 0

        public init(_ url: URL) throws {
            FileManager.default.createFile(atPath: url.path, contents: header(dataBytes: 0))
            handle = try FileHandle(forWritingTo: url)
        }

        deinit { try? handle.close() }

        public func append(_ samples: [Float]) throws {
            guard !samples.isEmpty else { return }
            let pcm = samples.map { Int16((max(-1, min(1, $0)) * 32767).rounded()).littleEndian }
            try handle.seekToEnd()
            try handle.write(contentsOf: pcm.withUnsafeBytes { Data($0) })
            dataBytes += UInt32(2 * pcm.count)
            try handle.seek(toOffset: 0)
            try handle.write(contentsOf: header(dataBytes: dataBytes))
        }
    }
}

private func header(dataBytes: UInt32) -> Data {
    var data = Data()
    let u32 = { (v: UInt32) in withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let u16 = { (v: UInt16) in withUnsafeBytes(of: v.littleEndian) { data.append(contentsOf: $0) } }
    let rate = UInt32(Audio.sampleRate)
    data.append(contentsOf: Array("RIFF".utf8)); u32(36 + dataBytes)
    data.append(contentsOf: Array("WAVEfmt ".utf8)); u32(16)
    u16(1); u16(1); u32(rate); u32(2 * rate); u16(2); u16(16)  // PCM, mono, 16-bit
    data.append(contentsOf: Array("data".utf8)); u32(dataBytes)
    return data
}
