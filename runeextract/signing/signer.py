"""Document signing and verification using Ed25519 (via cryptography) and SHA-256 hashing."""

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SigningError(Exception):
    """Raised when document signing fails."""


class VerificationError(Exception):
    """Raised when document verification fails."""


@dataclass
class SignatureInfo:
    algorithm: str = "SHA256-Ed25519"
    hash: str = ""
    signed_fields: list = field(default_factory=list)
    timestamp: Optional[str] = None
    public_key_fingerprint: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "hash": self.hash,
            "signed_fields": self.signed_fields,
            "timestamp": self.timestamp,
            "public_key_fingerprint": self.public_key_fingerprint,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SignatureInfo":
        return cls(
            algorithm=data.get("algorithm", "SHA256-Ed25519"),
            hash=data.get("hash", ""),
            signed_fields=data.get("signed_fields", []),
            timestamp=data.get("timestamp"),
            public_key_fingerprint=data.get("public_key_fingerprint", ""),
            metadata=data.get("metadata", {}),
        )


def compute_document_hash(doc: Any, fields: Optional[list] = None) -> str:
    hasher = hashlib.sha256()
    text = getattr(doc, "text", "")
    hasher.update(text.encode("utf-8"))
    if fields is None:
        fields = ["text", "tables", "source_type", "source_path"]
    for field_name in fields:
        val = getattr(doc, field_name, None)
        if val is not None:
            hasher.update(json.dumps(val, sort_keys=True, default=str).encode("utf-8"))
    return hasher.hexdigest()


def sign_document(doc: Any, private_key: bytes, fields: Optional[list] = None) -> SignatureInfo:
    try:
        from cryptography.hazmat.primitives import hashes as crypto_hashes
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives.serialization import load_der_private_key
    except ImportError as e:
        raise SigningError("cryptography library required for signing — pip install runeextract[signing]") from e
    try:
        if isinstance(private_key, bytes) and len(private_key) == 32:
            private_key_obj = ed25519.Ed25519PrivateKey.from_private_bytes(private_key)
        else:
            private_key_obj = load_der_private_key(private_key, password=None)
    except Exception as e:
        raise SigningError(f"Failed to load private key: {e}") from e

    doc_hash = compute_document_hash(doc, fields)
    try:
        signature_bytes = private_key_obj.sign(doc_hash.encode("utf-8"))
    except Exception as e:
        raise SigningError(f"Failed to sign: {e}") from e

    public_key = private_key_obj.public_key()
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
    pub_key_der = public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    fingerprint = hashlib.sha256(pub_key_der).hexdigest()[:16]

    info = SignatureInfo(
        hash=doc_hash,
        signed_fields=fields or ["text", "tables", "source_type", "source_path"],
        timestamp=datetime.now(timezone.utc).isoformat(),
        public_key_fingerprint=fingerprint,
    )
    setattr(doc, "_signature_info", info)
    setattr(doc, "_signature_bytes", signature_bytes.hex())
    return info


def verify_document(doc: Any, public_key: bytes, signature: Optional[str] = None) -> bool:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives.serialization import load_der_public_key
    except ImportError as e:
        raise VerificationError("cryptography library required for verification — pip install runeextract[signing]") from e
    try:
        if isinstance(public_key, bytes) and len(public_key) == 32:
            public_key_obj = ed25519.Ed25519PublicKey.from_public_bytes(public_key)
        else:
            public_key_obj = load_der_public_key(public_key)
    except Exception as e:
        raise VerificationError(f"Failed to load public key: {e}") from e

    if signature is None:
        signature = getattr(doc, "_signature_bytes", None)
    if signature is None:
        sig_info = getattr(doc, "_signature_info", None)
        if sig_info is not None:
            doc_hash = sig_info.hash
        else:
            raise VerificationError("No signature found on document — sign it first or pass signature explicitly")
    else:
        doc_hash = compute_document_hash(doc)
    signature_bytes = bytes.fromhex(signature) if isinstance(signature, str) else signature
    try:
        public_key_obj.verify(signature_bytes, doc_hash.encode("utf-8"))
        return True
    except Exception:
        return False


def generate_signing_keypair() -> tuple:
    try:
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:
        raise SigningError("cryptography library required") from e
    private_key = ed25519.Ed25519PrivateKey.generate()
    public_key = private_key.public_key()
    from cryptography.hazmat.primitives.serialization import Encoding, PrivateFormat, PublicFormat, NoEncryption
    priv_bytes = private_key.private_bytes(Encoding.DER, PrivateFormat.PKCS8, NoEncryption())
    pub_bytes = public_key.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
    return priv_bytes, pub_bytes


class DocumentSigner:
    def __init__(self, private_key: Optional[bytes] = None):
        if private_key is None:
            private_key, self._public_key = generate_signing_keypair()
        self._private_key = private_key

    def sign(self, doc: Any, fields: Optional[list] = None) -> SignatureInfo:
        return sign_document(doc, self._private_key, fields=fields)

    @property
    def public_key(self) -> bytes:
        if hasattr(self, "_public_key") and self._public_key is not None:
            return self._public_key
        from cryptography.hazmat.primitives.asymmetric import ed25519
        from cryptography.hazmat.primitives.serialization import load_der_private_key
        key = load_der_private_key(self._private_key, password=None)
        pub = key.public_key()
        from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
        self._public_key = pub.public_bytes(Encoding.DER, PublicFormat.SubjectPublicKeyInfo)
        return self._public_key


class DocumentVerifier:
    def __init__(self, public_key: Optional[bytes] = None):
        self._public_key = public_key

    def verify(self, doc: Any, signature: Optional[str] = None) -> bool:
        if self._public_key is None:
            raise VerificationError("No public key provided to verifier")
        return verify_document(doc, self._public_key, signature=signature)

    @staticmethod
    def verify_from_key(doc: Any, public_key: bytes, signature: Optional[str] = None) -> bool:
        return verify_document(doc, public_key, signature=signature)
