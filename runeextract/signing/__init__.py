"""Document signing & verification — hash integrity and Ed25519 signatures."""

from runeextract.signing.signer import (
    DocumentSigner,
    DocumentVerifier,
    sign_document,
    verify_document,
    compute_document_hash,
    SignatureInfo,
    SigningError,
    VerificationError,
)

__all__ = [
    "DocumentSigner",
    "DocumentVerifier",
    "sign_document",
    "verify_document",
    "compute_document_hash",
    "SignatureInfo",
    "SigningError",
    "VerificationError",
]
