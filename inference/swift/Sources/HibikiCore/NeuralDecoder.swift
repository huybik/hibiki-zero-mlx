import CoreML
import Foundation

/// Mimi's streaming decoder on the Neural Engine: the compiled Core ML model that
/// `inference/tools/export_mimi_coreml.py` writes from `mimi_step.DecoderStep`, one frame per
/// call in fp16. The model's state (convolution context and the transformer's key/value
/// windows) comes back as outputs and goes into the next call, and the attention mask and RoPE
/// tables are computed here from the step count, as `mimi_step.Inputs.at` does.
final class NeuralDecoder {
    /// Codebooks per frame, from the model's input shape.
    let codebooks: Int
    private let model: MLModel
    /// Transformer steps per call, steps in the key/value window, half a head's width.
    private let steps: Int
    private let window: Int
    private let halfDim: Int
    /// A new session's state; Core ML never writes to its inputs.
    private let zeros: [String: MLMultiArray]
    private var state: [String: MLMultiArray]
    /// Transformer steps run this session, two per frame.
    private var offset = 0

    init(url: URL) throws {
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        model = try MLModel(contentsOf: url, configuration: config)
        let shapes = model.modelDescription.inputDescriptionsByName.compactMapValues {
            $0.multiArrayConstraint?.shape.map(\.intValue)
        }
        let outputs = model.modelDescription.outputDescriptionsByName
        guard let codes = shapes["codes"], codes.count == 3, codes[2] == 1,
            let keys = shapes["keys"], keys.count == 4, let cos = shapes["cos"], cos.count == 2,
            ["conv", "values", "mask", "sin"].allSatisfy({ shapes[$0] != nil }),
            ["pcm", "new_conv", "new_keys", "new_values"].allSatisfy({
                outputs[$0]?.multiArrayConstraint?.dataType == .float16
            })
        else { throw HibikiError.unsupported("Mimi decoder model \(url.path)") }
        codebooks = codes[1]
        (steps, window, halfDim) = (cos[0], keys[2], cos[1])
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

    /// 1920 samples at 24 kHz from one frame of target codes.
    func decode(_ target: [Int]) throws -> [Float] {
        // Core ML returns autoreleased objects, which a thread without a run loop keeps
        // until it ends.
        try autoreleasepool { try predict(target) }
    }

    private func predict(_ target: [Int]) throws -> [Float] {
        precondition(target.count == codebooks)
        let codes = try MLMultiArray(shape: [1, codebooks, 1].map(NSNumber.init), dataType: .int32)
        let ints = codes.dataPointer.bindMemory(to: Int32.self, capacity: codes.count)
        for (q, code) in target.enumerated() { ints[q] = Int32(code) }
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
        offset += steps
        let pcm = out.featureValue(for: "pcm")!.multiArrayValue!
        let stride = pcm.strides.last!.intValue
        return pcm.withUnsafeBufferPointer(ofType: Float16.self) { values in
            (0..<pcm.count).map { Float(values[$0 * stride]) }
        }
    }

    /// The additive mask over [window, new steps] and the RoPE tables of this call's steps.
    /// Window slot j holds step `offset - window + j`, and the new steps follow it; a query at
    /// step s sees steps s - window ... s that exist.
    private func windowInputs() throws -> (MLMultiArray, MLMultiArray, MLMultiArray) {
        let mask = try MLMultiArray(shape: [steps, window + steps].map(NSNumber.init), dataType: .float16)
        let cos = try MLMultiArray(shape: [steps, halfDim].map(NSNumber.init), dataType: .float16)
        let sin = try MLMultiArray(shape: [steps, halfDim].map(NSNumber.init), dataType: .float16)
        let m = mask.dataPointer.bindMemory(to: Float16.self, capacity: mask.count)
        let c = cos.dataPointer.bindMemory(to: Float16.self, capacity: cos.count)
        let s = sin.dataPointer.bindMemory(to: Float16.self, capacity: sin.count)
        for i in 0..<steps {
            let query = offset + i
            for j in 0..<(window + steps) {
                let key = offset - window + j
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
