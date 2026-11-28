import Accelerate
import CoreML
import Foundation

/// The host's side of a `NeuralLM` frame: the trunk's inputs (the summed embeddings, RoPE tables,
/// cache slot and mask), the heads' and depformer's input rows, the depformer's Gumbel noise, and
/// reading Core ML's fp16 outputs.
extension NeuralLM {
    /// `temp * Gumbel(0, 1)` at 65,536 evenly spaced quantiles, in fp16 (128 KB): the table
    /// `fillGumbelNoise` draws from.
    static func gumbelTable(temp: Float) -> [Float16] {
        let t = Double(temp)
        return (0..<65536).map { i in
            let u = (Double(i) + 0.5) / 65536.0
            return Float16(-t * Foundation.log(-Foundation.log(u)))
        }
    }

    /// The summed input embeddings, the RoPE tables of this frame's position, its cache slot,
    /// and the slot it writes unmasked while the ring fills.
    func fillTrunkInputs(text textIn: Int, audio: [Int]) {
        let dim = cfg.dim
        precondition(textIn >= 0 && textIn <= cfg.textCard, "text token \(textIn)")
        var rowsIn = [textIn]
        for (q, token) in audio.enumerated() {
            precondition(token >= 0 && token <= cfg.card, "audio token \(token)")
            rowsIn.append(cfg.textCard + 1 + q * (cfg.card + 1) + token)
        }
        // Summed in fp32 (Accelerate), as fp16 rows from the table.
        var sum = [Float](repeating: 0, count: dim)
        var row = [Float](repeating: 0, count: dim)
        tables.withUnsafeBytes { raw in
            for index in rowsIn {
                row.withUnsafeMutableBytes { dst in
                    var src = vImage_Buffer(
                        data: UnsafeMutableRawPointer(mutating: raw.baseAddress! + index * dim * 2),
                        height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
                    var out = vImage_Buffer(
                        data: dst.baseAddress!, height: 1, width: vImagePixelCount(dim),
                        rowBytes: dim * 4)
                    vImageConvert_Planar16FtoPlanarF(&src, &out, 0)
                }
                sum.withUnsafeMutableBufferPointer { s in
                    vDSP_vadd(s.baseAddress!, 1, row, 1, s.baseAddress!, 1, vDSP_Length(dim))
                }
            }
        }
        sum.withUnsafeMutableBytes { src in
            var input = vImage_Buffer(
                data: src.baseAddress!, height: 1, width: vImagePixelCount(dim), rowBytes: dim * 4)
            var output = vImage_Buffer(
                data: x.dataPointer, height: 1, width: vImagePixelCount(dim), rowBytes: dim * 2)
            vImageConvert_PlanarFtoPlanar16F(&input, &output, 0)
        }

        let half = cos.count
        let c = cos.dataPointer.bindMemory(to: Float16.self, capacity: half)
        let s = sin.dataPointer.bindMemory(to: Float16.self, capacity: half)
        for i in 0..<half {
            // Positions grow without bound; the angle is computed in Double.
            let angle = Double(frame) * exp(-log(cfg.maxPeriod) * Double(i) / Double(half))
            c[i] = Float16(Foundation.cos(angle))
            s[i] = Float16(Foundation.sin(angle))
        }
        slot.dataPointer.bindMemory(to: Int32.self, capacity: 1)[0] = Int32(frame % cfg.context)
        if frame < cfg.context {
            mask.dataPointer.bindMemory(to: Float16.self, capacity: mask.count)[frame] = 0
        }
    }

    /// Row `row` of an fp16 `table` of `width` columns into `array` [1, width].
    func copyRow(_ row: Int, of table: Data, width: Int, into array: MLMultiArray) {
        let dst = array.dataPointer.bindMemory(to: Float16.self, capacity: width)
        table.withUnsafeBytes { raw in
            let src = raw.bindMemory(to: Float16.self).baseAddress! + row * width
            dst.update(from: src, count: width)
        }
    }

    /// Fills `noise` [dep_q, card] fp16 with i.i.d. `temp * Gumbel(0, 1)` from `gumbelLUT`.
    func fillGumbelNoise(into noise: MLMultiArray) {
        let n = noise.count
        let dst = noise.dataPointer.bindMemory(to: Float16.self, capacity: n)
        gumbelLUT.withUnsafeBufferPointer { lut in
            let table = lut.baseAddress!
            var i = 0
            while i + 3 < n {
                let r = rng.next()
                dst[i] = table[Int(r & 0xFFFF)]
                dst[i + 1] = table[Int((r >> 16) & 0xFFFF)]
                dst[i + 2] = table[Int((r >> 32) & 0xFFFF)]
                dst[i + 3] = table[Int(r >> 48)]
                i += 4
            }
            while i < n {
                dst[i] = table[Int(rng.next() & 0xFFFF)]
                i += 1
            }
        }
    }

    /// Reads exact integer token IDs from a 1-D fp16 Core ML output `[count]`.
    func readTokens(_ array: MLMultiArray, count: Int) -> [Int] {
        let stride = array.strides[0].intValue
        return array.withUnsafeBufferPointer(ofType: Float16.self) { values in
            (0..<count).map { i in
                let tok = Int(Float(values[i * stride]).rounded())
                precondition(tok >= 0 && tok < cfg.card, "depformer token \(tok)")
                return tok
            }
        }
    }

    /// A 2-D fp16 output's rows as Float, by its strides (Core ML may pad rows).
    func rows(_ array: MLMultiArray) -> [[Float]] {
        let shape = array.shape.map(\.intValue)
        let strides = array.strides.map(\.intValue)
        precondition(shape.count == 2 && array.dataType == .float16)
        return array.withUnsafeBufferPointer(ofType: Float16.self) { values in
            (0..<shape[0]).map { r in
                [Float](unsafeUninitializedCapacity: shape[1]) { row, count in
                    let base = values.baseAddress! + r * strides[0]
                    for c in 0..<shape[1] { row[c] = Float(base[c * strides[1]]) }
                    count = shape[1]
                }
            }
        }
    }
}
