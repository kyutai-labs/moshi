// MoshiFFI.swift
// Idiomatic Swift wrapper for the Moshi C FFI layer.
//
// This file wraps the C functions exported by the moshi-ios Rust crate
// (via MoshiFFI.xcframework) into a safe, ergonomic Swift API with:
// - ARC-based memory management (deinit calls moshi_destroy)
// - Thread safety via a private serial DispatchQueue
// - Swift Error bridging from moshi_get_last_error()
// - No force-unwraps; all optionals handled safely

import Foundation
import MoshiFFI  // C module from XCFramework (moshi_ios.h + module.modulemap)

/// Thread-safe Swift wrapper around a Moshi inference session.
///
/// All FFI calls are dispatched on a private serial queue to prevent
/// data races, since the underlying Rust `MoshiSession` is not Send/Sync.
///
/// Usage:
/// ```swift
/// let session = try MoshiSession()
/// try session.loadModel(lmPath: "/path/to/model.gguf", mimiPath: "/path/to/mimi.safetensors")
/// let output = try session.processAudio(input: inputSamples)
/// ```
public final class MoshiSession {

    // MARK: - Constants

    /// Audio sample rate in Hz (24,000).
    public static let sampleRate: Int = Int(moshi_get_sample_rate())

    /// Number of PCM samples per audio frame (1920 samples = 80ms at 24kHz).
    public static let frameSamples: Int = 1920

    /// Whether Metal GPU acceleration is available on this device.
    public static var isMetalAvailable: Bool {
        return moshi_metal_available() == 1
    }

    // MARK: - Private state

    /// Opaque pointer to the Rust MoshiSession.
    private var handle: OpaquePointer?

    /// Serial queue enforcing single-threaded access to FFI calls.
    private let queue = DispatchQueue(label: "ai.openclaw.moshi.ffi", qos: .userInitiated)

    // MARK: - Init / Deinit

    /// Create a new Moshi session backed by Metal.
    ///
    /// - Throws: `MoshiError.sessionCreationFailed` if the Metal device
    ///   cannot be initialized or session allocation fails.
    public init() throws {
        let ptr = moshi_create()
        guard let ptr = ptr else {
            let message = Self.consumeLastError() ?? "unknown error"
            throw MoshiError.sessionCreationFailed(message)
        }
        self.handle = OpaquePointer(ptr)
    }

    deinit {
        if let handle = handle {
            // Convert OpaquePointer back to the typed pointer for moshi_destroy.
            let typed = UnsafeMutablePointer<moshi_session_t>(handle)
            moshi_destroy(typed)
        }
    }

    // MARK: - Model Loading

    /// Load the LM and Mimi models from file paths.
    ///
    /// - Parameters:
    ///   - lmPath: Absolute path to the LM model file (e.g. model.q8.gguf).
    ///   - mimiPath: Absolute path to the Mimi codec model (e.g. model.safetensors).
    /// - Throws: `MoshiError.loadFailed` if model loading fails.
    public func loadModel(lmPath: String, mimiPath: String) throws {
        try performSync { handle in
            let result = lmPath.withCString { lm in
                mimiPath.withCString { mimi in
                    moshi_load_model_with_assets(handle, lm, mimi)
                }
            }
            if result != 0 {
                let message = Self.consumeLastError() ?? "unknown load error"
                throw MoshiError.loadFailed(message)
            }
        }
    }

    /// Load the LM, Mimi, and SentencePiece tokenizer models from file paths.
    ///
    /// - Parameters:
    ///   - lmPath: Absolute path to the LM model file (e.g. model.q8.gguf).
    ///   - mimiPath: Absolute path to the Mimi codec model (e.g. model.safetensors).
    ///   - tokenizerPath: Absolute path to the SentencePiece tokenizer model.
    ///     Pass `nil` to skip tokenizer loading (text output will be unavailable).
    /// - Throws: `MoshiError.loadFailed` if model loading fails.
    public func loadModel(lmPath: String, mimiPath: String, tokenizerPath: String?) throws {
        try performSync { handle in
            let result: Int32
            if let tokenizerPath = tokenizerPath {
                result = lmPath.withCString { lm in
                    mimiPath.withCString { mimi in
                        tokenizerPath.withCString { tok in
                            moshi_load_model_with_assets_and_tokenizer(handle, lm, mimi, tok)
                        }
                    }
                }
            } else {
                result = lmPath.withCString { lm in
                    mimiPath.withCString { mimi in
                        moshi_load_model_with_assets_and_tokenizer(handle, lm, mimi, nil)
                    }
                }
            }
            if result != 0 {
                let message = Self.consumeLastError() ?? "unknown load error"
                throw MoshiError.loadFailed(message)
            }
        }
    }

    // MARK: - Audio Processing

    /// Process a frame of input PCM audio and return the model's output audio.
    ///
    /// Input and output are mono Float32 PCM at 24kHz. The expected input
    /// frame size is ``frameSamples`` (1920 samples = 80ms).
    ///
    /// - Parameter input: Array of Float32 PCM samples. May be empty, in which
    ///   case an empty array is returned (not an error).
    /// - Returns: Array of Float32 PCM output samples. May be empty if the
    ///   codec has not yet accumulated enough data.
    /// - Throws: `MoshiError.notInitialized` if no model is loaded.
    ///           `MoshiError.processingFailed` if inference fails.
    public func processAudio(input: [Float]) -> Result<[Float], MoshiError> {
        return performSyncResult { handle in
            // Fast path: empty input returns empty output (matches Rust behavior).
            if input.isEmpty {
                return .success([])
            }

            // Allocate output buffer. The output length is at most the input
            // length for normal operation, but we add headroom.
            let outputCapacity = max(input.count * 2, Self.frameSamples * 2)
            var outputBuffer = [Float](repeating: 0, count: outputCapacity)
            var outputLen: Int32 = 0

            let result = input.withUnsafeBufferPointer { inBuf in
                outputBuffer.withUnsafeMutableBufferPointer { outBuf in
                    moshi_process_audio_ex(
                        handle,
                        inBuf.baseAddress,
                        Int32(input.count),
                        outBuf.baseAddress,
                        Int32(outputCapacity),
                        &outputLen
                    )
                }
            }

            if result != 0 {
                let message = Self.consumeLastError() ?? "unknown processing error"
                return .failure(.processingFailed(message))
            }

            let count = Int(outputLen)
            if count == 0 {
                return .success([])
            }
            return .success(Array(outputBuffer.prefix(count)))
        }
    }

    // MARK: - Text Output

    /// Get model-generated text accumulated since the last call.
    ///
    /// Returns `nil` if no new text is available. The string memory is
    /// managed internally using `moshi_free_string`.
    ///
    /// - Returns: The latest model text, or `nil`.
    public func getModelText() -> String? {
        return performSyncOptional { handle in
            let ptr = moshi_get_model_text(handle)
            guard let ptr = ptr else {
                return nil
            }
            defer { moshi_free_string(ptr) }
            return String(cString: ptr)
        }
    }

    /// Get user-transcribed text accumulated since the last call.
    ///
    /// Returns `nil` if no new text is available. Note: this may always
    /// return `nil` in the current implementation (pending user text
    /// transcription is not yet populated by the Rust backend).
    ///
    /// - Returns: The latest user text, or `nil`.
    public func getUserText() -> String? {
        return performSyncOptional { handle in
            let ptr = moshi_get_user_text(handle)
            guard let ptr = ptr else {
                return nil
            }
            defer { moshi_free_string(ptr) }
            return String(cString: ptr)
        }
    }

    // MARK: - Session Management

    /// Reset the streaming inference state without unloading model weights.
    ///
    /// Call this between conversations to clear accumulated text and
    /// reset the codec/LM state. Models remain loaded and ready.
    public func reset() {
        performSyncVoid { handle in
            moshi_reset(handle)
        }
    }

    /// Whether the session has models loaded and is ready for inference.
    public var isInitialized: Bool {
        return performSyncValue { handle in
            return moshi_is_initialized(handle) == 1
        }
    }

    /// The sample rate configured for this session, in Hz.
    public var sessionSampleRate: Int {
        return performSyncValue { handle in
            return Int(moshi_session_sample_rate(handle))
        }
    }

    // MARK: - Private Helpers

    /// Consume the last FFI error string, returning nil if none is set.
    private static func consumeLastError() -> String? {
        let ptr = moshi_get_last_error()
        guard let ptr = ptr else {
            return nil
        }
        defer { moshi_free_string(ptr) }
        return String(cString: ptr)
    }

    /// Get the typed pointer from the opaque handle, throwing if nil.
    private func typedHandle() throws -> UnsafeMutablePointer<moshi_session_t> {
        guard let handle = handle else {
            throw MoshiError.notInitialized
        }
        return UnsafeMutablePointer<moshi_session_t>(handle)
    }

    /// Execute a throwing closure on the serial queue with the typed FFI handle.
    private func performSync(_ body: (UnsafeMutablePointer<moshi_session_t>) throws -> Void) throws {
        try queue.sync {
            let typed = try typedHandle()
            try body(typed)
        }
    }

    /// Execute a closure returning a Result on the serial queue.
    private func performSyncResult<T>(
        _ body: (UnsafeMutablePointer<moshi_session_t>) -> Result<T, MoshiError>
    ) -> Result<T, MoshiError> {
        return queue.sync {
            guard let typed = try? typedHandle() else {
                return .failure(.notInitialized)
            }
            return body(typed)
        }
    }

    /// Execute a closure returning an optional on the serial queue.
    private func performSyncOptional<T>(
        _ body: (UnsafeMutablePointer<moshi_session_t>) -> T?
    ) -> T? {
        return queue.sync {
            guard let typed = try? typedHandle() else {
                return nil
            }
            return body(typed)
        }
    }

    /// Execute a closure returning a value on the serial queue.
    ///
    /// If the handle is nil, the C API functions accept null session pointers
    /// and return sensible defaults (e.g. sample rate, 0 for isInitialized).
    private func performSyncValue<T>(
        _ body: (UnsafeMutablePointer<moshi_session_t>?) -> T
    ) -> T {
        return queue.sync {
            guard let handle = handle else {
                return body(nil)
            }
            let typed = UnsafeMutablePointer<moshi_session_t>(handle)
            return body(typed)
        }
    }

    /// Execute a void closure on the serial queue.
    private func performSyncVoid(
        _ body: (UnsafeMutablePointer<moshi_session_t>) -> Void
    ) {
        queue.sync {
            guard let typed = try? typedHandle() else {
                return
            }
            body(typed)
        }
    }
}
