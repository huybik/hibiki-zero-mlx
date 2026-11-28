import SwiftUI

struct ContentView: View {
    @EnvironmentObject private var controller: InferenceController

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            header
            modelPicker
            languagePicker
            fileControls
            microphoneControls
            transcript
            backendLog
        }
        .padding(22)
        .frame(minWidth: 720, minHeight: 620)
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 3) {
                Text("Hibiki Test")
                    .font(.largeTitle.bold())
                Text("Vietnamese ↔ English speech and text (the model hears which)")
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Text(controller.status)
                .font(.callout.weight(.semibold))
                .foregroundStyle(controller.isRunning ? .orange : .secondary)
                .padding(.horizontal, 11)
                .padding(.vertical, 6)
                .background(.quaternary, in: Capsule())
        }
    }

    private var modelPicker: some View {
        GroupBox("MLX model") {
            HStack(spacing: 12) {
                Image(systemName: "cpu")
                    .foregroundStyle(.secondary)
                Text(controller.modelURL?.path ?? "Choose a staged model directory")
                    .lineLimit(1)
                    .truncationMode(.middle)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .textSelection(.enabled)
                Button("Choose…", action: controller.chooseModel)
                    .disabled(controller.isRunning)
            }
            .padding(.vertical, 4)
        }
    }

    private var languagePicker: some View {
        GroupBox("Spoken language") {
            HStack(spacing: 12) {
                Picker("Spoken language", selection: $controller.speaking) {
                    Text("Auto").tag(String?.none)
                    Text("Vietnamese").tag(String?.some("vi"))
                    Text("English").tag(String?.some("en"))
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .fixedSize()
                .disabled(controller.isRunning)
                Text("Auto lets the model guess. If it answers in the language you speak, pick that language.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.vertical, 4)
        }
    }

    private var fileControls: some View {
        GroupBox("Translate an audio file") {
            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    Image(systemName: "waveform")
                        .foregroundStyle(.secondary)
                    Text(controller.audioURL?.path ?? "Choose a WAV, MP3, or FLAC file")
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .textSelection(.enabled)
                    Button("Choose audio…", action: controller.chooseAudio)
                        .disabled(controller.isRunning)
                }
                HStack {
                    if controller.isFileRunning {
                        Button(action: controller.stop) {
                            Label("Stop", systemImage: "stop.fill")
                        }
                        .buttonStyle(.borderedProminent)
                        .tint(.red)
                    } else {
                        Button(action: controller.translateFile) {
                            Label("Translate file", systemImage: "character.bubble")
                        }
                        .buttonStyle(.borderedProminent)
                        .disabled(!controller.canTranslateFile)
                    }

                    Button(action: controller.playOutput) {
                        Label("Play output", systemImage: "play.fill")
                    }
                    .disabled(controller.outputAudioURL == nil || controller.isRunning)

                    Button(action: controller.revealOutput) {
                        Label("Show in Finder", systemImage: "folder")
                    }
                    .disabled(controller.outputAudioURL == nil)
                }
            }
            .padding(.vertical, 4)
        }
    }

    private var microphoneControls: some View {
        GroupBox("Live microphone") {
            HStack {
                if controller.isMicrophoneRunning {
                    Button(action: controller.stop) {
                        Label("Stop microphone", systemImage: "stop.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(.red)
                } else {
                    Button(action: controller.startMicrophone) {
                        Label("Start microphone", systemImage: "mic.fill")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(controller.isRunning || controller.modelURL == nil)
                }
                Text("Uses the current macOS input and output devices.")
                    .font(.callout)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            .padding(.vertical, 4)
        }
    }

    private var transcript: some View {
        GroupBox("Translation") {
            ScrollView {
                Text(controller.transcript.isEmpty ? "Translation will appear here." : controller.transcript)
                    .foregroundStyle(controller.transcript.isEmpty ? .secondary : .primary)
                    .font(.system(.body, design: .rounded))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(8)
            }
            .frame(minHeight: 120)
        }
    }

    private var backendLog: some View {
        DisclosureGroup("Backend log") {
            ScrollView {
                Text(controller.log.isEmpty ? "No backend output yet." : controller.log)
                    .font(.system(.caption, design: .monospaced))
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .topLeading)
            }
            .frame(height: 72)
            .padding(.top, 6)
        }
    }
}
