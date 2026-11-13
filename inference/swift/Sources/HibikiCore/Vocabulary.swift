import Foundation

/// Token pieces of a SentencePiece `.model`, read straight from its protobuf: `ModelProto`
/// field 1 holds the `SentencePiece` messages (piece = 1, score = 2, type = 3).
public struct Vocabulary {
    public let pieces: [String]
    /// `<unk>`, `</s>` and `<pad>`: never printed, and silence for the stop rule
    /// (`text_special_ids` in pipeline.py).
    public let special: Set<Int>
    public let eos: Int
    /// The byte each byte-fallback piece (type BYTE, `<0xE1>`) stands for.
    let bytes: [Int: UInt8]

    public init(url: URL) throws {
        let data = [UInt8](try Data(contentsOf: url))
        var pieces: [String] = []
        var special = Set<Int>()
        var eos: Int?
        var bytes: [Int: UInt8] = [:]
        var reader = ProtoReader(data[...])
        while let (field, wire) = try reader.key() {
            guard field == 1, wire == 2 else {
                try reader.skip(wire)
                continue
            }
            var message = ProtoReader(try reader.bytes())
            var piece = ""
            var type = 1  // NORMAL
            while let (f, w) = try message.key() {
                switch (f, w) {
                case (1, 2): piece = String(decoding: try message.bytes(), as: UTF8.self)
                case (3, 0): type = Int(try message.varint())
                default: try message.skip(w)
                }
            }
            if type == 2 || piece == "</s>" || piece == "<pad>" { special.insert(pieces.count) }
            if piece == "</s>" { eos = pieces.count }
            if type == 6 {
                guard piece.hasPrefix("<0x"), piece.hasSuffix(">"),
                    let byte = UInt8(piece.dropFirst(3).dropLast(), radix: 16)
                else { throw HibikiError.badTokenizer("byte piece \(piece) in \(url.path)") }
                bytes[pieces.count] = byte
            }
            pieces.append(piece)
        }
        guard let eos else { throw HibikiError.badTokenizer("no </s> piece in \(url.path)") }
        self.pieces = pieces
        self.special = special
        self.eos = eos
        self.bytes = bytes
    }

    /// The UTF-8 a token adds to the text: its piece with "▁" as a space, its byte for a
    /// byte-fallback piece, nothing for a special token.
    func utf8(_ token: Int) -> [UInt8] {
        if special.contains(token) { return [] }
        if let byte = bytes[token] { return [byte] }
        return Array(pieces[token].replacingOccurrences(of: "▁", with: " ").utf8)
    }
}

/// The text of a stream of text tokens, as `TextDecoder` in pipeline.py: for valid UTF-8, what
/// the tokenizer decodes from the stream's non-special tokens. A byte-fallback piece carries one
/// UTF-8 byte, so a character the vocabulary lacks spans several tokens, which may fall in
/// different steps: its text comes out once its sequence is complete, an invalid run of bytes as
/// one U+FFFD, and `flush` returns what is left when the stream ends. Special tokens add nothing
/// and do not break a sequence.
struct TextDecoder {
    let vocabulary: Vocabulary
    /// The start of a UTF-8 sequence that later tokens may complete.
    private var pending: [UInt8] = []

    init(_ vocabulary: Vocabulary) { self.vocabulary = vocabulary }

    /// The text `tokens` complete.
    mutating func callAsFunction(_ tokens: [Int]) -> String {
        for token in tokens { pending += vocabulary.utf8(token) }
        let complete = Self.completeCount(pending)
        defer { pending.removeFirst(complete) }
        return String(decoding: pending[..<complete], as: UTF8.self)
    }

    /// What is left when the stream ends: a sequence the model never completed is U+FFFD.
    mutating func flush() -> String {
        defer { pending = [] }
        return String(decoding: pending, as: UTF8.self)
    }

    /// The number of leading bytes that later bytes cannot change: all but a trailing lead byte
    /// with fewer continuation bytes than it announces.
    static func completeCount(_ bytes: [UInt8]) -> Int {
        var lead = bytes.count - 1
        while lead >= 0, bytes.count - lead <= 3, bytes[lead] & 0xC0 == 0x80 { lead -= 1 }
        guard lead >= 0 else { return bytes.count }
        let length: Int
        switch bytes[lead] {
        case 0xC2...0xDF: length = 2
        case 0xE0...0xEF: length = 3
        case 0xF0...0xF4: length = 4
        default: return bytes.count
        }
        return bytes.count - lead < length ? lead : bytes.count
    }
}

private struct ProtoReader {
    var data: ArraySlice<UInt8>

    init(_ data: ArraySlice<UInt8>) { self.data = data }

    mutating func varint() throws -> UInt64 {
        var value: UInt64 = 0
        var shift: UInt64 = 0
        while let byte = data.popFirst() {
            value |= UInt64(byte & 0x7f) << shift
            if byte & 0x80 == 0 { return value }
            shift += 7
        }
        throw HibikiError.badTokenizer("truncated varint")
    }

    mutating func key() throws -> (Int, Int)? {
        if data.isEmpty { return nil }
        let key = try varint()
        return (Int(key >> 3), Int(key & 7))
    }

    mutating func bytes() throws -> ArraySlice<UInt8> {
        let count = Int(try varint())
        guard count <= data.count else { throw HibikiError.badTokenizer("truncated field") }
        let value = data.prefix(count)
        data = data.dropFirst(count)
        return value
    }

    mutating func skip(_ wire: Int) throws {
        switch wire {
        case 0: _ = try varint()
        case 1: data = data.dropFirst(8)
        case 2: _ = try bytes()
        case 5: data = data.dropFirst(4)
        default: throw HibikiError.badTokenizer("wire type \(wire)")
        }
    }
}
