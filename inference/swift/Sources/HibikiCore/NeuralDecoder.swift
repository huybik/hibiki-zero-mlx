import CoreML
import Foundation

/// Mimi's streaming decoder on the Neural Engine: the Core ML package that
/// `inference/tools/export_mimi_coreml.py` writes from `mimi_step.DecoderStep`. Each call
/// decodes a fixed number of frames in fp16; the model's state (convolution context and the
/// transformer's key/value windows) comes back as outputs and goes into the next call, and the
/// attention mask and RoPE tables are computed here from the step count, as
/// `mimi_step.Inputs.at` does.
final class NeuralDecoder {
    /// Frames per call and codebooks per frame, from the package's input shapes.
    let frames: Int
    let codebooks: Int
    private let model: MLModel
    private let shapes: [String: [Int]]
    /// A new session's state; Core ML never writes to its inputs.
    private let zeros: [String: MLMultiArray]
    private var state: [String: MLMultiArray]
    /// Transformer steps run this session, two per frame.
    private var offset = 0

    init(url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: MLModel.compileModel(at: url), configuration: config)
        let shapes = model.modelDescription.inputDescriptionsByName.compactMapValues {
            $0.multiArrayConstraint?.shape.map(\.intValue)
        }
        guard let codes = shapes["codes"], codes.count == 3,
            ["conv", "keys", "values", "mask", "cos", "sin"].allSatisfy({ shapes[$0] != nil })
        else { throw HibikiError.unsupported("Mimi decoder package \(url.path)") }
        self.shapes = shapes
        codebooks = codes[1]
        frames = codes[2]
        zeros = try Dictionary(uniqueKeysWithValues: ["conv", "keys", "values"].map { name in
            let array = try MLMultiArray(shape: shapes[name]!.map(NSNumber.init), dataType: .float16)
            memset(array.dataPointer, 0, array.count * MemoryLayout<Float16>.size)
            return (name, array)
        })
        state = zeros
    }

    /// Starts a new session: zero convolution context, empty key/value windows.
    func reset() {
        offset = 0
        state = zeros
    }

    /// 1920 samples at 24 kHz for each of exactly `frames` frames of target codes.
    func decode(_ targets: [[Int]]) throws -> [Float] {
        // Core ML returns autoreleased objects, which a thread without a run loop keeps
        // until it ends.
        try autoreleasepool { try predict(targets) }
    }

    private func predict(_ targets: [[Int]]) throws -> [Float] {
        precondition(targets.count == frames && targets.allSatisfy { $0.count == codebooks })
        let codes = try MLMultiArray(shape: [1, codebooks, frames].map(NSNumber.init), dataType: .int32)
        let ints = codes.dataPointer.bindMemory(to: Int32.self, capacity: codes.count)
        for (f, target) in targets.enumerated() {
            for (q, code) in target.enumerated() { ints[q * frames + f] = Int32(code) }
        }
        let (mask, cos, sin) = try windowInputs()
        var inputs = state.mapValues { MLFeatureValue(multiArray: $0) }
        inputs["codes"] = MLFeatureValue(multiArray: codes)
        inputs["mask"] = MLFeatureValue(multiArray: mask)
        inputs["cos"] = MLFeatureValue(multiArray: cos)
        inputs["sin"] = MLFeatureValue(multiArray: sin)
        let out = try model.prediction(from: MLDictionaryFeatureProvider(dictionary: inputs))
        for name in ["conv", "keys", "values"] {
            state[name] = out.featureValue(for: "new_\(name)")!.multiArrayValue!
        }
        offset += 2 * frames
        let pcm = out.featureValue(for: "pcm")!.multiArrayValue!
        let stride = pcm.strides.last!.intValue
        return pcm.withUnsafeBufferPointer(ofType: Float16.self) { values in
            (0..<pcm.count).map { Float(values[$0 * stride]) }
        }
    }

    /// The additive mask over [window, new steps] and the RoPE tables of this call's steps.
    /// Window slot j holds step `offset - window + j`; a query at step s sees steps
    /// s - window ... s that exist.
    private func windowInputs() throws -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        let (steps, window, halfDim) = (shapes["cos"]![0], shapes["keys"]![2], shapes["cos"]![1])
        let mask = try MLMultiArray(shape: [steps, window + steps].map(NSNumber.init), dataType: .float16)
        let cos = try MLMultiArray(shape: [steps, halfDim].map(NSNumber.init), dataType: .float16)
        let sin = try MLMultiArray(shape: [steps, halfDim].map(NSNumber.init), dataType: .float16)
        let m = mask.dataPointer.bindMemory(to: Float16.self, capacity: mask.count)
        let c = cos.dataPointer.bindMemory(to: Float16.self, capacity: cos.count)
        let s = sin.dataPointer.bindMemory(to: Float16.self, capacity: sin.count)
        for i in 0..<steps {
            let query = offset + i
            for j in 0..<(window + steps) {
                let key = j < window ? offset - window + j : offset + j - window
                let visible = key >= 0 && query - key >= 0 && query - key <= window
                m[i * (window + steps) + j] = visible ? 0 : -1e4
            }
            for d in 0..<halfDim {
                // Positions grow without bound; the angle is computed in Double.
                let angle = Double(query) * exp(Double(d) * -log(10000.0) * 2 / Double(2 * halfDim))
                c[i * halfDim + d] = Float16(Foundation.cos(angle))
                s[i * halfDim + d] = Float16(Foundation.sin(angle))
            }
        }
        return (mask, cos, sin)
    }
}
