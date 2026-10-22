// swift-tools-version: 5.9

import PackageDescription

let package = Package(
    name: "HibikiTestApp",
    platforms: [.macOS(.v14)],
    targets: [
        .executableTarget(
            name: "HibikiTestApp",
            path: "Sources/HibikiTestApp"
        )
    ]
)
