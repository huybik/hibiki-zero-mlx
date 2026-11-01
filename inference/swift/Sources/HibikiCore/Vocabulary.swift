import Foundation

/// Token pieces of a SentencePiece `.model`, read straight from its protobuf: `ModelProto`
/// field 1 holds the `SentencePiece` messages (piece = 1, score = 2, type = 3).
public struct Vocabulary {
    public let pieces: [String]
    /// Control and unknown pieces (`<unk>`, `<s>`, `</s>`, `<pad>`), never printed.
    public let special: Set<Int>
    public let eos: Int

    public init(url: URL) throws {
        let data = [UInt8](try Data(contentsOf: url))
        var pieces: [String] = []
        var special = Set<Int>()
        var eos: Int?
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
            if type == 2 || type == 3 { special.insert(pieces.count) }  // UNKNOWN, CONTROL
            if piece == "</s>" { eos = pieces.count }
            pieces.append(piece)
        }
        guard let eos else { throw HibikiError.badTokenizer("no </s> piece in \(url.path)") }
        self.pieces = pieces
        self.special = special
        self.eos = eos
    }

    /// The text a token adds to the transcript, or nil for special tokens.
    public func text(_ token: Int) -> String? {
        special.contains(token) ? nil : pieces[token].replacingOccurrences(of: "▁", with: " ")
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
