import { useRef, useCallback, useState } from "react";
import OpusRecorder from "opus-recorder";

const getAudioWorkletNode = async (
  audioContext: AudioContext,
  name: string,
) => {
  try {
    return new AudioWorkletNode(audioContext, name);
  } catch {
    await audioContext.audioWorklet.addModule(`/${name}.js`);
    return new AudioWorkletNode(audioContext, name, {});
  }
};

export interface AudioProcessor {
  audioContext: AudioContext;
  opusRecorder: OpusRecorder;
  decoder: DecoderWorker;
  outputWorklet: AudioWorkletNode;
  inputAnalyser: AnalyserNode;
  outputAnalyser: AnalyserNode;
  mediaStreamDestination: MediaStreamAudioDestinationNode;
  mediaRecorder: MediaRecorder;
  // processingDelaySec: number;
}

type WorkletStats = {
  totalAudioPlayed: number;
  actualAudioPlayed: number;
  minDelay: number;
  maxDelay: number;
};

export const useAudioProcessor = (
  onOpusRecorded: (chunk: Uint8Array) => void,
) => {
  const audioProcessorRef = useRef<AudioProcessor | null>(null);
  const [processingDelaySec, setProcessingDelaySec] = useState(0);
  const recordedChunksRef = useRef<Blob[]>([]);
  const [hasRecording, setHasRecording] = useState(false);

  const setupAudio = useCallback(
    async (mediaStream: MediaStream) => {
      if (audioProcessorRef.current) return audioProcessorRef.current;

      const audioContext = new AudioContext();
      const outputWorklet = await getAudioWorkletNode(
        audioContext,
        "audio-output-processor",
      );
      const source = audioContext.createMediaStreamSource(mediaStream);
      // source.connect(inputWorklet);
      const inputAnalyser = audioContext.createAnalyser();
      inputAnalyser.fftSize = 2048;
      source.connect(inputAnalyser);

      // Stereo recording: model=left (channel 0), user=right (channel 1)
      const merger = audioContext.createChannelMerger(2);
      outputWorklet.connect(merger, 0, 0); // model → left
      source.connect(merger, 0, 1); // user → right
      const mediaStreamDestination =
        audioContext.createMediaStreamDestination();
      merger.connect(mediaStreamDestination);

      outputWorklet.connect(audioContext.destination);
      const outputAnalyser = audioContext.createAnalyser();
      outputAnalyser.fftSize = 2048;
      outputWorklet.connect(outputAnalyser);

      const decoder = new Worker("/decoderWorker.min.js");
      let micDuration = 0;

      outputWorklet.port.onmessage = (event: MessageEvent<WorkletStats>) => {
        const curDelay =
          event.data.totalAudioPlayed - event.data.actualAudioPlayed;
        setProcessingDelaySec(curDelay);
      };

      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      decoder.onmessage = (event: MessageEvent<any>) => {
        if (!event.data) {
          return;
        }
        const frame = event.data[0];
        outputWorklet.port.postMessage({
          frame: frame,
          type: "audio",
          micDuration: micDuration,
        });
      };
      decoder.postMessage({
        command: "init",
        bufferLength: (960 * audioContext.sampleRate) / 24000,
        decoderSampleRate: 24000,
        outputBufferSampleRate: audioContext.sampleRate,
        resampleQuality: 0,
      });

      // For buffer length: 960 = 24000 / 12.5 / 2
      // The /2 is a bit optional, but won't hurt for recording the mic.
      // Note that bufferLength actually has 0 impact for mono audio, only
      // the frameSize and maxFramesPerPage seems to have any.
      const recorderOptions = {
        mediaTrackConstraints: {
          audio: {
            echoCancellation: true,
            noiseSuppression: false,
            autoGainControl: true,
            channelCount: 1,
          },
          video: false,
        },
        encoderPath: "/encoderWorker.min.js",
        bufferLength: Math.round((960 * audioContext.sampleRate) / 24000),
        encoderFrameSize: 20,
        encoderSampleRate: 24000,
        maxFramesPerPage: 2,
        numberOfChannels: 1,
        recordingGain: 1,
        resampleQuality: 3,
        encoderComplexity: 0,
        encoderApplication: 2049,
        streamPages: true,
      };
      let chunk_idx = 0;
      let lastpos = 0;
      const opusRecorder = new OpusRecorder(recorderOptions);
      opusRecorder.ondataavailable = (data: Uint8Array) => {
        // opus actually always works at 48khz, so it seems this is the proper value to use here.
        micDuration = opusRecorder.encodedSamplePosition / 48000;
        // logging disabled
        if (chunk_idx < 0) {
          console.debug(
            Date.now() % 1000,
            "Mic Data chunk",
            chunk_idx++,
            (opusRecorder.encodedSamplePosition - lastpos) / 48000,
            micDuration,
          );
          lastpos = opusRecorder.encodedSamplePosition;
        }
        onOpusRecorded(data);
      };
      // Set up stereo recording via MediaRecorder
      recordedChunksRef.current = [];
      setHasRecording(false);
      const mediaRecorder = new MediaRecorder(mediaStreamDestination.stream);
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      mediaRecorder.onstop = () => {
        setHasRecording(recordedChunksRef.current.length > 0);
      };

      audioProcessorRef.current = {
        audioContext,
        opusRecorder,
        decoder,
        outputWorklet,
        inputAnalyser,
        outputAnalyser,
        mediaStreamDestination,
        mediaRecorder,
      };
      // Resume the audio context if it was suspended
      audioProcessorRef.current.audioContext.resume();
      opusRecorder.start();
      mediaRecorder.start(1000);

      return audioProcessorRef.current;
    },
    [onOpusRecorded],
  );

  const shutdownAudio = useCallback(() => {
    if (audioProcessorRef.current) {
      const { audioContext, opusRecorder, outputWorklet, mediaRecorder } =
        audioProcessorRef.current;

      // Stop the stereo recorder first
      if (mediaRecorder.state !== "inactive") {
        mediaRecorder.stop();
      }

      // Disconnect all nodes
      outputWorklet.disconnect();
      audioContext.close();
      opusRecorder.stop();

      // Clear the reference
      audioProcessorRef.current = null;
    }
  }, []);

  const getRecordingBlob = useCallback(() => {
    const mimeType =
      audioProcessorRef.current?.mediaRecorder.mimeType || "audio/webm";
    return new Blob(recordedChunksRef.current, { type: mimeType });
  }, []);

  return {
    setupAudio,
    shutdownAudio,
    audioProcessor: audioProcessorRef,
    processingDelaySec,
    hasRecording,
    getRecordingBlob,
  };
};
