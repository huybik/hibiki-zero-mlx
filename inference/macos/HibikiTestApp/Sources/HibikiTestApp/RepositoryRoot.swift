import Foundation

extension InferenceController {
    /// The checkout this app runs: the first directory holding `inference/main.py` and
    /// `inference/hibiki_mlx/`, walking up from the current directory, then from the app's bundle.
    static func findRepositoryRoot() -> URL? {
        let fileManager = FileManager.default
        let starts = [
            URL(fileURLWithPath: fileManager.currentDirectoryPath, isDirectory: true),
            Bundle.main.bundleURL,
        ]
        for start in starts {
            var candidate = start
            for _ in 0..<8 {
                let main = candidate.appendingPathComponent("inference/main.py")
                let runtime = candidate.appendingPathComponent(
                    "inference/hibiki_mlx", isDirectory: true
                )
                var isDirectory: ObjCBool = false
                if fileManager.fileExists(atPath: main.path),
                   fileManager.fileExists(atPath: runtime.path, isDirectory: &isDirectory),
                   isDirectory.boolValue {
                    return candidate
                }
                candidate.deleteLastPathComponent()
            }
        }
        return nil
    }
}
