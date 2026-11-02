// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "HibikiSwift",
    platforms: [.iOS(.v18), .macOS(.v15)],
    products: [
        .library(name: "HibikiCore", targets: ["HibikiCore"]),
        .executable(name: "hibiki", targets: ["hibiki"]),
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift", exact: "0.32.3"),
    ],
    targets: [
        .target(
            name: "HibikiCore",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"),
            ]
        ),
        .executableTarget(name: "hibiki", dependencies: ["HibikiCore"]),
    ],
    swiftLanguageModes: [.v5]
)
