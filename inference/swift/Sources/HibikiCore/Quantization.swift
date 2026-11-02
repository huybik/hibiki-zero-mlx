// Copyright (c) Kyutai, all rights reserved.
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

import MLX
import MLXNN

class EuclideanCodebook: Module {
    let epsilon: Float
    let dim: Int
    // Derived from the loaded weights; the leading underscore keeps them out of the parameters.
    var _embedding: MLXArray
    var _c2: MLXArray
    @ModuleInfo(key: "_initialized") var initialized: MLXArray
    @ModuleInfo(key: "embedding_sum") var embeddingSum: MLXArray
    @ModuleInfo(key: "cluster_usage") var clusterUsage: MLXArray

    init(dim: Int, codebookSize: Int) {
        self.epsilon = 1e-5
        self.dim = dim
        self._initialized.wrappedValue = MLXArray.zeros([1], dtype: .float32)
        self._embeddingSum.wrappedValue = MLXArray.zeros([codebookSize, dim], dtype: .float32)
        self._clusterUsage.wrappedValue = MLXArray.zeros([codebookSize], dtype: .float32)
        let clusterUsage = maximum(self._clusterUsage.wrappedValue, self.epsilon)[0..., .newAxis]
        self._embedding = self._embeddingSum.wrappedValue / clusterUsage
        self._c2 = self._embedding.square().sum(axis: -1) / 2
    }

    override func update(
        parameters: ModuleParameters, verify: Module.VerifyUpdate, path: [String] = [],
        modulePath: [String] = []
    ) throws -> Self {
        try super.update(
            parameters: parameters, verify: verify, path: path, modulePath: modulePath)
        let clusterUsage = maximum(self._clusterUsage.wrappedValue, self.epsilon)[0..., .newAxis]
        self._embedding = self._embeddingSum.wrappedValue / clusterUsage
        self._c2 = self._embedding.square().sum(axis: -1) / 2
        return self
    }

    func encode(_ x: MLXArray) -> MLXArray {
        let targetShape = Array(x.shape.dropLast())
        let x = x.flattened(end: -2)
        let dotProd = x.matmul(_embedding.swappedAxes(-1, -2))
        return (_c2 - dotProd).argMin(axis: -1).reshaped(targetShape)
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        let finalDims = indexes.shape + [self.dim]
        let indexes = indexes.flattened()
        return _embedding.take(indexes, axis: 0).reshaped(finalDims)
    }
}

class VectorQuantization: Module {
    @ModuleInfo(key: "_codebook") var codebook: EuclideanCodebook

    init(dim: Int, codebookSize: Int) {
        self._codebook.wrappedValue = EuclideanCodebook(dim: dim, codebookSize: codebookSize)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        self.codebook.encode(x.swappedAxes(-1, -2))
    }

    func decode(_ indexes: MLXArray) -> MLXArray {
        self.codebook.decode(indexes).swappedAxes(-1, -2)
    }
}

class ResidualVectorQuantization: Module {
    @ModuleInfo(key: "layers") var layers: [VectorQuantization]

    init(nQ: Int, dim: Int, codebookSize: Int) {
        self._layers.wrappedValue = (0..<nQ).map { _ in
            VectorQuantization(dim: dim, codebookSize: codebookSize)
        }
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var codes: [MLXArray] = []
        var residual = x
        for layer in self.layers {
            let indices = layer.encode(residual)
            let quantized = layer.decode(indices)
            residual = residual - quantized
            codes.append(indices)
        }
        return stacked(codes, axis: 0)
    }

}

class ResidualVectorQuantizer: Module {
    @ModuleInfo(key: "vq") var vq: ResidualVectorQuantization
    @ModuleInfo(key: "input_proj") var inputProj: Conv1d

    init(dim: Int, inputDim: Int, nQ: Int, bins: Int) {
        self._inputProj.wrappedValue = Conv1d(
            inputChannels: inputDim, outputChannels: dim, kernelSize: 1, bias: false)
        self._vq.wrappedValue = ResidualVectorQuantization(nQ: nQ, dim: dim, codebookSize: bins)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        self.vq.encode(inputProj(x)).swappedAxes(0, 1)
    }
}

class SplitResidualVectorQuantizer: Module {
    let nQ: Int
    @ModuleInfo(key: "rvq_first") var rvqFirst: ResidualVectorQuantizer
    @ModuleInfo(key: "rvq_rest") var rvqRest: ResidualVectorQuantizer

    init(dim: Int, inputDim: Int, nQ: Int, bins: Int) {
        self.nQ = nQ
        self._rvqFirst.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, nQ: 1, bins: bins)
        self._rvqRest.wrappedValue = ResidualVectorQuantizer(
            dim: dim, inputDim: inputDim, nQ: nQ - 1, bins: bins)
    }

    func encode(_ x: MLXArray) -> MLXArray {
        var codes = self.rvqFirst.encode(x)
        if self.nQ > 1 {
            let restCodes = self.rvqRest.encode(x)
            codes = concatenated([codes, restCodes], axis: 1)
        }
        return codes
    }
}
