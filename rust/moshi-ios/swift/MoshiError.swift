// MoshiError.swift
// Error types for the Moshi FFI Swift wrapper.

import Foundation

/// Errors that can occur during Moshi FFI operations.
public enum MoshiError: Error, LocalizedError {
    /// Model loading failed with the given message from the Rust FFI layer.
    case loadFailed(String)

    /// Audio processing or inference failed with the given message.
    case processingFailed(String)

    /// The session has not been initialized (no models loaded).
    case notInitialized

    /// Session creation failed with the given message.
    case sessionCreationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .loadFailed(let message):
            return "Moshi model load failed: \(message)"
        case .processingFailed(let message):
            return "Moshi processing failed: \(message)"
        case .notInitialized:
            return "Moshi session is not initialized (no model loaded)"
        case .sessionCreationFailed(let message):
            return "Moshi session creation failed: \(message)"
        }
    }
}
