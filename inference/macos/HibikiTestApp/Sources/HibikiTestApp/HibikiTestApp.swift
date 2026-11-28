import AppKit
import SwiftUI

@main
struct HibikiTestApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @StateObject private var controller = InferenceController()

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environmentObject(controller)
        }
        .defaultSize(width: 780, height: 680)
    }
}

/// One window: closing it quits, which ends the backend, rather than leaving a mic session running
/// with no window to stop it.
final class AppDelegate: NSObject, NSApplicationDelegate {
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        true
    }
}
