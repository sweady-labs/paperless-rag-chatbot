from hashlib import sha256


def stable_int_id(key: str) -> int:
    """Return a stable positive integer id from a key string using SHA256 (63-bit).

    This avoids using Python's built-in `hash()` which is randomized per process.
    """
    return int(sha256(key.encode('utf-8')).hexdigest(), 16) & 0x7FFFFFFFFFFFFFFF
