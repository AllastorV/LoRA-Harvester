"""
Character Recognizer - InsightFace based face recognition with hybrid matching
Supports:
  - Reference-based matching (supervised): match faces to known characters
  - Auto-clustering (unsupervised): group unknown faces automatically
  - Hybrid mode: reference match first, cluster the rest
"""

import os
import cv2
import json
import time
import shutil
import sqlite3
import logging
import threading
import numpy as np
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Iterator, List, Optional, Tuple
from concurrent.futures import Future, ThreadPoolExecutor

logger = logging.getLogger(__name__)

# Embedding dimension produced by all InsightFace `buffalo_*` / `antelopev2`
# recognition models.
_EMBED_DIM = 512

# Lazy imports to avoid hard crash if not installed
_insightface = None
_sklearn = None


def _imread_unicode(path) -> Optional[np.ndarray]:
    """
    Read an image robustly, including from paths containing non-ASCII
    characters on Windows (where ``cv2.imread`` silently returns None
    because it uses a narrow-char C string).

    Returns the decoded BGR image, or None on failure.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.debug("Failed to read image %s: %s", path, e)
        return None


FaceList = List[Tuple[np.ndarray, float]]  # [(embedding[512], bbox_area), ...]


class EmbeddingCache:
    """
    SQLite-backed cache of face detection results keyed by
    ``(absolute path, mtime_ns, size)``. Invalidated automatically when the
    source file is modified.

    Storage format per row:
      * ``n_faces`` faces, each with a 512-D float32 embedding + one float32
        bbox area, packed contiguously as ``(n_faces, 513)`` and stored as a
        raw BLOB. Zero-face rows (no face detected) store an empty BLOB and
        are still cached — this is important so that empty images don't get
        re-inferred on every run.

    The cache is process-wide thread safe (serialises writes through a
    mutex) and uses WAL + NORMAL sync for fast concurrent access.
    """

    _SCHEMA_VERSION = 2

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS embeddings (
        path     TEXT NOT NULL,
        model    TEXT NOT NULL,
        mtime_ns INTEGER NOT NULL,
        size     INTEGER NOT NULL,
        n_faces  INTEGER NOT NULL,
        data     BLOB    NOT NULL,
        PRIMARY KEY (path, model)
    );
    """

    def __init__(self, path: Path, model_name: str = "buffalo_l"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self._lock = threading.Lock()
        # Autocommit mode — we manage BEGIN/COMMIT explicitly around
        # batches. Passing isolation_level=None *without* explicit
        # transactions would silently defeat the periodic commit() below
        # (each execute would commit on its own).
        self._conn = sqlite3.connect(
            str(self.path),
            check_same_thread=False,
            isolation_level=None,
        )
        # Performance pragmas — small risk of losing the last few inserts on
        # a hard crash, which is acceptable for a cache that can be rebuilt.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA temp_store=MEMORY")

        # Schema migration: if the stored schema version is older than the
        # current one, drop the old table (cache entries can always be
        # rebuilt from the source images). Handles the v1 → v2 upgrade
        # where the primary key changed from (path) to (path, model).
        row = self._conn.execute("PRAGMA user_version").fetchone()
        current_version = row[0] if row else 0
        if current_version < self._SCHEMA_VERSION:
            self._conn.execute("DROP TABLE IF EXISTS embeddings")
            self._conn.execute("DROP INDEX IF EXISTS ix_embeddings_model")
            self._conn.execute(f"PRAGMA user_version = {self._SCHEMA_VERSION}")
        self._conn.executescript(self._SCHEMA)

        self._pending = 0
        self._in_tx = False
        self.hits = 0
        self.misses = 0
        self._dirty = False

    # ── single-row ops ────────────────────────────────────────────────────

    def get(self, file_path: Path) -> Optional[FaceList]:
        """Return cached faces or None on miss / stale entry."""
        try:
            stat = os.stat(str(file_path))
        except OSError:
            return None
        key = str(Path(file_path).resolve())
        with self._lock:
            row = self._conn.execute(
                "SELECT mtime_ns, size, n_faces, data FROM embeddings "
                "WHERE path=? AND model=?",
                (key, self.model_name),
            ).fetchone()
        if row is None:
            self.misses += 1
            return None
        mtime_ns, size, n_faces, data = row
        if mtime_ns != stat.st_mtime_ns or size != stat.st_size:
            self.misses += 1
            return None
        self.hits += 1
        if n_faces == 0:
            return []
        arr = np.frombuffer(data, dtype=np.float32)
        arr = arr.reshape(n_faces, _EMBED_DIM + 1).copy()  # copy → writable
        return [(arr[i, :_EMBED_DIM], float(arr[i, _EMBED_DIM])) for i in range(n_faces)]

    def put(self, file_path: Path, faces: FaceList) -> None:
        """Store (or replace) faces for ``file_path``."""
        try:
            stat = os.stat(str(file_path))
        except OSError:
            return
        key = str(Path(file_path).resolve())
        if faces:
            packed = np.zeros((len(faces), _EMBED_DIM + 1), dtype=np.float32)
            for i, (emb, area) in enumerate(faces):
                packed[i, :_EMBED_DIM] = emb
                packed[i, _EMBED_DIM] = area
            blob = packed.tobytes()
        else:
            blob = b""
        with self._lock:
            if not self._in_tx:
                self._conn.execute("BEGIN")
                self._in_tx = True
            self._conn.execute(
                "INSERT OR REPLACE INTO embeddings "
                "(path, model, mtime_ns, size, n_faces, data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (key, self.model_name, stat.st_mtime_ns, stat.st_size, len(faces), blob),
            )
            self._pending += 1
            self._dirty = True
            if self._pending >= 64:
                self._conn.execute("COMMIT")
                self._in_tx = False
                self._pending = 0

    def flush(self) -> None:
        with self._lock:
            if self._in_tx:
                self._conn.execute("COMMIT")
                self._in_tx = False
                self._pending = 0

    def close(self) -> None:
        with self._lock:
            try:
                if self._in_tx:
                    self._conn.execute("COMMIT")
                    self._in_tx = False
                    self._pending = 0
            finally:
                self._conn.close()

    def stats(self) -> Dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}


# ─────────────────────────────────────────────────────────────────────────────
# Parallel image decoding
# ─────────────────────────────────────────────────────────────────────────────

def _prefetch_decode(
    paths: Iterable[Path],
    workers: int = 4,
    max_inflight: int = 16,
) -> Iterator[Tuple[Path, Optional[np.ndarray]]]:
    """
    Yield ``(path, decoded_image_or_None)`` pairs in the original order while
    decoding images concurrently in the background.

    InsightFace inference itself must run on a single thread (ONNXRuntime
    sessions are not safe to call from multiple Python threads
    simultaneously), but image decoding is pure CPU and releases the GIL
    inside OpenCV — so running the decode in a small thread pool lets us
    overlap disk I/O + JPEG decoding with GPU inference on the next item.
    """
    paths = list(paths)
    if not paths:
        return

    if workers <= 1 or len(paths) == 1:
        for p in paths:
            yield p, _imread_unicode(p)
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        it = iter(paths)
        # deque gives O(1) popleft vs O(N) list.pop(0) — matters when
        # max_inflight is large and/or we're processing tens of thousands
        # of images.
        pending: Deque[Tuple[Path, "Future[Optional[np.ndarray]]"]] = deque()

        # Prime the pipeline
        for _ in range(max_inflight):
            try:
                p = next(it)
            except StopIteration:
                break
            pending.append((p, ex.submit(_imread_unicode, p)))

        while pending:
            p, fut = pending.popleft()
            try:
                img = fut.result()
            except Exception as e:
                logger.warning("Decode failed for %s: %s", p, e)
                img = None
            yield p, img
            try:
                next_p = next(it)
                pending.append((next_p, ex.submit(_imread_unicode, next_p)))
            except StopIteration:
                pass


def _get_insightface():
    global _insightface
    if _insightface is None:
        try:
            import insightface
            from insightface.app import FaceAnalysis
            _insightface = FaceAnalysis
        except ImportError:
            raise ImportError(
                "InsightFace is not installed.\n"
                "Install with: pip install insightface onnxruntime"
            )
    return _insightface


def _get_dbscan():
    global _sklearn
    if _sklearn is None:
        try:
            from sklearn.cluster import DBSCAN
            _sklearn = DBSCAN
        except ImportError:
            raise ImportError(
                "scikit-learn is not installed.\n"
                "Install with: pip install scikit-learn"
            )
    return _sklearn


# ─────────────────────────────────────────────────────────────────────────────
# Core class
# ─────────────────────────────────────────────────────────────────────────────

class CharacterRecognizer:
    """
    Hybrid character recognizer:
      1. Tries to match against reference images (if provided)
      2. Clusters unmatched faces automatically
      3. Frames with no detected face go to 'no_face/' folder
    """

    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    def __init__(
        self,
        reference_dir: Optional[str] = None,
        similarity_threshold: float = 0.45,
        match_margin: float = 0.05,
        cluster_eps: float = 0.6,
        cluster_min_samples: int = 2,
        use_gpu: bool = True,
        model_name: str = "buffalo_l",
        progress_callback=None,
        cache_path: Optional[str] = None,
        use_cache: bool = True,
        num_workers: int = 4,
    ):
        """
        Args:
            reference_dir: Path to folder containing sub-folders per character.
                           E.g. references/naruto/*.jpg  references/sasuke/*.jpg
            similarity_threshold: Cosine distance threshold for reference matching.
                                  Lower = stricter. Typical range 0.3-0.6.
            match_margin: Minimum gap required between the best and the
                          second-best reference match. If two references are
                          too close (distance difference < margin) the result
                          is considered ambiguous and rejected.
            cluster_eps: DBSCAN eps parameter for clustering unknowns.
            cluster_min_samples: DBSCAN min_samples parameter.
            use_gpu: Use GPU if available.
            model_name: InsightFace model pack ('buffalo_l' recommended).
            progress_callback: Optional callable(current, total, message).
            cache_path: Path to the embedding cache SQLite file. When None
                        (and ``use_cache`` is True) a cache is created inside
                        the directory being processed. Set ``use_cache=False``
                        to disable caching entirely.
            use_cache: Enable/disable the persistent embedding cache.
            num_workers: Number of background threads used to decode images
                        in parallel. Set to 1 to disable prefetching.
        """
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.similarity_threshold = similarity_threshold
        self.match_margin = match_margin
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.progress_callback = progress_callback
        self.num_workers = max(1, int(num_workers))
        self.use_cache = use_cache
        self._cache_path_override = cache_path
        self._cache: Optional[EmbeddingCache] = None

        # {character_name: np.ndarray of shape (N, 512)}
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        # Mean embedding per character (kept for backwards compatibility;
        # matching now prefers the full per-sample distance, see
        # ``match_to_reference``).
        self.reference_means: Dict[str, np.ndarray] = {}

        # Vectorised reference matching state — populated by load_references
        # and used by match_to_reference. Storing the stacked matrix once lets
        # us do one matmul per query instead of one per character.
        self._ref_matrix: Optional[np.ndarray] = None  # (N_total, 512)
        self._ref_char_names: List[str] = []           # per-character order
        self._ref_char_slices: List[Tuple[int, int]] = []  # (start, end) per char

        self._app = None  # InsightFace FaceAnalysis (lazy)
        self._app_lock = threading.Lock()  # guard lazy model init across threads
        # InsightFace sessions are not safe to call from multiple Python
        # threads simultaneously, so serialise inference even if the caller
        # fans out.
        self._inference_lock = threading.Lock()

    # ─── InsightFace initialization ───────────────────────────────────────────

    def _load_model(self):
        if self._app is not None:
            return
        with self._app_lock:
            if self._app is not None:
                return
            FaceAnalysis = _get_insightface()

            # Explicitly set ONNXRuntime providers so GPU preference is honoured
            # and users get a clear log line if CUDA is unavailable and we
            # silently fall back to CPU.
            if self.use_gpu:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                try:
                    import onnxruntime as ort
                    available = set(ort.get_available_providers())
                    if 'CUDAExecutionProvider' not in available:
                        logger.warning(
                            "CUDAExecutionProvider not available in onnxruntime "
                            "(available: %s). Falling back to CPU. Install "
                            "onnxruntime-gpu for GPU acceleration.",
                            sorted(available),
                        )
                        providers = ['CPUExecutionProvider']
                except ImportError:
                    logger.warning(
                        "onnxruntime not importable — provider selection skipped."
                    )
            else:
                providers = ['CPUExecutionProvider']

            ctx_id = 0 if providers[0] == 'CUDAExecutionProvider' else -1
            from src.core.model_paths import INSIGHTFACE_DIR, ensure_dirs
            ensure_dirs()
            app = FaceAnalysis(
                name=self.model_name,
                root=str(INSIGHTFACE_DIR),
                allowed_modules=['detection', 'recognition'],
                providers=providers,
            )
            app.prepare(ctx_id=ctx_id, det_size=(640, 640))
            self._app = app
            logger.info(
                "InsightFace model '%s' loaded (ctx_id=%d, providers=%s)",
                self.model_name, ctx_id, providers,
            )

    # ─── Embedding extraction ──────────────────────────────────────────────────

    @staticmethod
    def _normalise(emb: np.ndarray) -> np.ndarray:
        emb = emb.astype(np.float32)
        return emb / (np.linalg.norm(emb) + 1e-6)

    def _detect_faces(self, image: np.ndarray) -> FaceList:
        """
        Run InsightFace once and return [(normalised_embedding, bbox_area), ...]
        sorted by bbox area descending. This is the single entry point used by
        both ``get_embeddings`` and ``get_largest_face_embedding`` so we never
        run the detection pipeline twice on the same image.
        """
        self._load_model()
        with self._inference_lock:
            faces = self._app.get(image)  # BGR expected (OpenCV default)
        results: FaceList = []
        for face in faces:
            if face.embedding is None:
                continue
            bbox = face.bbox
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            results.append((self._normalise(face.embedding), area))
        # Sort descending by area so results[0] is always the largest face
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ─── Embedding cache helpers ─────────────────────────────────────────────

    def _open_cache(self, anchor_dir: Path) -> Optional[EmbeddingCache]:
        """
        Open (or re-use) the embedding cache. When no explicit cache path is
        configured the cache lives alongside the directory being processed so
        it's scoped per-dataset and trivially cleanable.

        If a cache is already open at a *different* path, it is closed and
        re-opened at the new anchor. This ensures that calling
        ``load_references()`` (anchored on the reference dir) followed by
        ``sort_directory()`` (anchored on the input dir) ends up using the
        dataset's own cache, not the reference directory's.
        """
        if not self.use_cache:
            return None
        if self._cache_path_override:
            target = Path(self._cache_path_override).resolve()
        else:
            target = (Path(anchor_dir) / ".lora_harvester_cache.db").resolve()

        if self._cache is not None:
            try:
                current = Path(self._cache.path).resolve()
            except OSError:
                current = Path(self._cache.path)
            if current == target:
                return self._cache
            try:
                self._cache.close()
            finally:
                self._cache = None

        try:
            self._cache = EmbeddingCache(target, model_name=self.model_name)
            logger.info("Embedding cache: %s", target)
        except sqlite3.Error as e:
            logger.warning("Could not open embedding cache at %s: %s", target, e)
            self._cache = None
        return self._cache

    def close_cache(self) -> None:
        if self._cache is not None:
            try:
                self._cache.close()
            finally:
                self._cache = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close_cache()

    def get_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect all faces in an image and return their L2-normalised embeddings.
        Returns empty list if no face detected.
        """
        return [emb for emb, _ in self._detect_faces(image)]

    def get_largest_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return embedding of the largest face only (most likely the main subject)."""
        faces = self._detect_faces(image)
        if not faces:
            return None
        return faces[0][0]

    # ─── Reference loading ────────────────────────────────────────────────────

    def _get_faces_for_path(
        self,
        path: Path,
        decoded: Optional[np.ndarray],
    ) -> Optional[FaceList]:
        """
        Return faces for ``path``, preferring the cache. ``decoded`` is the
        already-decoded image (from the prefetcher) or None if the caller
        didn't decode — useful when the cache hits and we can skip decode
        entirely on the fast path. Returns None on decode failure.
        """
        if self._cache is not None:
            cached = self._cache.get(path)
            if cached is not None:
                return cached
        # Cache miss: make sure we have pixels
        img = decoded if decoded is not None else _imread_unicode(path)
        if img is None:
            return None
        faces = self._detect_faces(img)
        if self._cache is not None:
            self._cache.put(path, faces)
        return faces

    def load_references(self, reference_dir: Optional[str] = None) -> Dict[str, int]:
        """
        Load reference images from sub-folders.

        Expected structure:
            references/
                character_a/
                    img1.jpg
                    img2.jpg
                character_b/
                    img1.jpg

        Returns:
            {character_name: number_of_embeddings_loaded}
        """
        ref_path = Path(reference_dir) if reference_dir else self.reference_dir
        if ref_path is None:
            logger.warning("No reference directory specified.")
            return {}

        if not ref_path.exists():
            raise FileNotFoundError(f"Reference directory not found: {ref_path}")

        self._load_model()
        self._open_cache(ref_path)
        counts: Dict[str, int] = {}

        char_dirs = [d for d in sorted(ref_path.iterdir()) if d.is_dir()]
        if not char_dirs:
            logger.warning("No character sub-folders found in %s", ref_path)
            return {}

        print(f"Loading references from: {ref_path}")

        # Reset previous state so back-to-back calls behave deterministically.
        self.reference_embeddings.clear()
        self.reference_means.clear()

        for char_dir in char_dirs:
            char_name = char_dir.name
            embeddings: List[np.ndarray] = []
            unreadable = 0
            no_face = 0
            multi_face_refs = 0

            image_files = [
                f for f in sorted(char_dir.iterdir())
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

            # Split into cache hits (no decode) vs misses (need prefetch).
            to_decode: List[Path] = []
            direct_hits: Dict[Path, FaceList] = {}
            for f in image_files:
                if self._cache is not None:
                    hit = self._cache.get(f)
                    if hit is not None:
                        direct_hits[f] = hit
                        continue
                to_decode.append(f)

            # Process cache hits first (trivial).
            for f in image_files:
                if f in direct_hits:
                    faces = direct_hits[f]
                    if not faces:
                        no_face += 1
                        continue
                    if len(faces) > 1:
                        multi_face_refs += 1
                    embeddings.append(faces[0][0])

            # Then prefetch-decode + infer the misses.
            for f, img in _prefetch_decode(to_decode, workers=self.num_workers):
                if img is None:
                    unreadable += 1
                    logger.warning("Could not read reference image: %s", f)
                    continue
                faces = self._detect_faces(img)
                if self._cache is not None:
                    try:
                        self._cache.put(f, faces)
                    except sqlite3.Error as e:
                        logger.warning("Cache put failed for %s: %s", f, e)
                if not faces:
                    no_face += 1
                    logger.warning("No face detected in reference image: %s", f)
                    continue
                if len(faces) > 1:
                    multi_face_refs += 1
                    logger.info(
                        "Reference %s contains %d faces; using only the largest. "
                        "Consider cropping to a single face for better accuracy.",
                        f, len(faces),
                    )
                embeddings.append(faces[0][0])

            if embeddings:
                emb_array = np.stack(embeddings)  # (N, 512)
                self.reference_embeddings[char_name] = emb_array
                mean = emb_array.mean(axis=0)
                self.reference_means[char_name] = mean / (np.linalg.norm(mean) + 1e-6)
                counts[char_name] = len(embeddings)
                extra = []
                if unreadable:
                    extra.append(f"{unreadable} unreadable")
                if no_face:
                    extra.append(f"{no_face} no-face")
                if multi_face_refs:
                    extra.append(f"{multi_face_refs} multi-face (used largest)")
                extra_str = f" ({', '.join(extra)})" if extra else ""
                print(f"  {char_name}: {len(embeddings)} reference(s) loaded{extra_str}")
            else:
                print(
                    f"  {char_name}: No usable references "
                    f"(unreadable={unreadable}, no_face={no_face}) — skipping"
                )

        # Build the stacked reference matrix once for fast matching.
        self._rebuild_reference_matrix()
        if self._cache is not None:
            self._cache.flush()
        return counts

    def _rebuild_reference_matrix(self) -> None:
        """
        Flatten per-character reference embeddings into a single (N_total, 512)
        matrix so ``match_to_reference`` can do one matmul per query instead
        of one per character. Also stores a parallel list of (start, end)
        slices per character name.
        """
        self._ref_char_names = []
        self._ref_char_slices = []
        rows: List[np.ndarray] = []
        cursor = 0
        for name, arr in self.reference_embeddings.items():
            n = arr.shape[0]
            if n == 0:
                continue
            rows.append(arr)
            self._ref_char_names.append(name)
            self._ref_char_slices.append((cursor, cursor + n))
            cursor += n
        if rows:
            # rows already come from np.stack on float32 embeddings, so
            # concatenate preserves the float32 dtype and C-contiguity.
            self._ref_matrix = np.concatenate(rows, axis=0)
        else:
            self._ref_matrix = None

    # ─── Matching ─────────────────────────────────────────────────────────────

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two normalised vectors. Range [0, 2]."""
        return float(1.0 - np.dot(a, b))

    def match_to_reference(self, embedding: np.ndarray) -> Optional[str]:
        """
        Match a query embedding to the closest reference character.

        Uses a single matmul against the pre-stacked reference matrix
        (O(total_refs) instead of one loop per character), then reduces per
        character by taking the max similarity within that character's slice.

        Returns the character name only when:
          - the best distance is within ``similarity_threshold``, AND
          - the gap to the second-best character is at least ``match_margin``.
        """
        if self._ref_matrix is None or not self._ref_char_names:
            return None

        sims = self._ref_matrix @ embedding  # (N_total,)

        # Reduce per character — these slices are tiny (one per character,
        # typically <10 characters) so a Python loop is cheaper than fancy
        # reduction tricks.
        per_char = np.empty(len(self._ref_char_names), dtype=np.float32)
        for i, (start, end) in enumerate(self._ref_char_slices):
            per_char[i] = sims[start:end].max()

        # Argsort descending (higher similarity first == lower distance).
        order = np.argsort(-per_char)
        best_idx = int(order[0])
        best_dist = float(1.0 - per_char[best_idx])

        if best_dist > self.similarity_threshold:
            return None

        if len(order) >= 2:
            runner_idx = int(order[1])
            runner_dist = float(1.0 - per_char[runner_idx])
            if (runner_dist - best_dist) < self.match_margin:
                logger.debug(
                    "Ambiguous match: %s@%.3f vs %s@%.3f (margin=%.3f)",
                    self._ref_char_names[best_idx], best_dist,
                    self._ref_char_names[runner_idx], runner_dist,
                    self.match_margin,
                )
                return None

        return self._ref_char_names[best_idx]

    # ─── Main sorting pipeline ────────────────────────────────────────────────

    # System folder names that are never counted toward the character limit
    _SYSTEM_FOLDERS = {"no_face", "multi_face", "unknown", "other", "trimmed"}

    def sort_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        copy: bool = False,
        recursive: bool = False,
        max_characters: int = 6,
        max_per_character: int = 0,
    ) -> Dict[str, int]:
        """
        Sort images in input_dir into character sub-folders.

        Mode:
          - If references loaded → tries reference matching first
          - Unmatched faces → auto-clustered into character_01/, character_02/ ...
          - No faces detected → no_face/
          - Multiple faces with ambiguous reference match → multi_face/
          - If more character folders than max_characters → smallest ones merged to other/

        Args:
            input_dir: Directory with images to sort.
            output_dir: Where to place sorted images.
                        If None, creates sub-folders inside input_dir.
            copy: Copy files instead of moving.
            recursive: Also scan sub-directories.
            max_characters: Maximum number of character folders to create (1-6).
                            Extra characters (by image count) go to other/.

        Returns:
            {folder_name: count} stats dict.
        """
        input_path = Path(input_dir).resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        if not input_path.is_dir():
            raise NotADirectoryError(f"Input path is not a directory: {input_dir}")

        out_path = (
            Path(output_dir).resolve() if output_dir
            else input_path / "_sorted"
        )

        # Safety: never allow the output to *equal* the input — that would
        # make the sort walk over its own destination and is almost certainly
        # a mistake. Always force an output sub-directory in that case.
        if out_path == input_path:
            out_path = input_path / "_sorted"
            logger.warning(
                "output_dir equals input_dir; redirecting output to %s", out_path,
            )

        out_path.mkdir(parents=True, exist_ok=True)

        # Fast, single-pass directory walk via os.scandir. For large datasets
        # this is noticeably faster than Path.glob("**/*") because it avoids
        # stat() calls and Path object overhead for every intermediate entry.
        image_files: List[Path] = self._scan_images(
            input_path, out_path, recursive,
        )

        if not image_files:
            print(f"No image files found in {input_dir}")
            return {}

        run_started = time.monotonic()
        self._load_model()
        self._open_cache(input_path)

        total = len(image_files)
        print(f"\nProcessing {total} images...")
        print(f"References loaded: {list(self.reference_embeddings.keys()) or 'None (auto-cluster only)'}")
        print(f"Similarity threshold: {self.similarity_threshold}  margin: {self.match_margin}")
        print(f"Workers: {self.num_workers}  Cache: {'on' if self._cache else 'off'}")

        # ── Pass 1: extract embeddings & do reference matching ──────────────
        unknown_items: List[Tuple[Path, np.ndarray]] = []   # (path, embedding)
        no_face_paths: List[Path] = []
        multi_face_paths: List[Path] = []
        assignments: Dict[Path, str] = {}  # path → character name

        # Split into cache hits vs misses. The fast path never touches the
        # disk for hits — we only prefetch-decode the actual misses.
        cache_hits: Dict[Path, FaceList] = {}
        to_decode: List[Path] = []
        if self._cache is not None:
            for f in image_files:
                hit = self._cache.get(f)
                if hit is not None:
                    cache_hits[f] = hit
                else:
                    to_decode.append(f)
        else:
            to_decode = list(image_files)

        # Run all face detections first, stashing into a dict keyed by path.
        faces_by_path: Dict[Path, FaceList] = dict(cache_hits)

        if to_decode:
            started = time.monotonic()
            for idx_decode, (img_path, img) in enumerate(
                _prefetch_decode(to_decode, workers=self.num_workers), 1,
            ):
                if self.progress_callback:
                    self.progress_callback(
                        idx_decode, len(to_decode),
                        f"Inferring {img_path.name}",
                    )
                if img is None:
                    logger.warning("Failed to decode image: %s", img_path)
                    faces_by_path[img_path] = []  # treat as "no face detected"
                    if self._cache is not None:
                        # Don't cache decode failures — they may be transient
                        pass
                    continue
                faces = self._detect_faces(img)
                faces_by_path[img_path] = faces
                if self._cache is not None:
                    try:
                        self._cache.put(img_path, faces)
                    except sqlite3.Error as e:
                        logger.warning("Cache put failed for %s: %s", img_path, e)
            elapsed = time.monotonic() - started
            if elapsed > 0 and len(to_decode) > 0:
                logger.info(
                    "Face inference: %d images in %.2fs (%.1f img/s)",
                    len(to_decode), elapsed, len(to_decode) / elapsed,
                )

        if self._cache is not None:
            self._cache.flush()
            logger.info(
                "Cache: %d hits, %d misses",
                self._cache.hits, self._cache.misses,
            )

        # Pass 1b: route every image using its (cached or fresh) face list.
        # Only emit routing progress when the decode phase did nothing
        # (i.e. everything was a cache hit) so we don't double-tick the bar.
        routing_progress = self.progress_callback is not None and not to_decode
        for idx, img_path in enumerate(image_files, 1):
            if routing_progress:
                self.progress_callback(idx, total, f"Sorting {img_path.name}")

            faces = faces_by_path.get(img_path, [])

            if not faces:
                no_face_paths.append(img_path)
                continue

            if len(faces) == 1:
                emb = faces[0][0]
                matched = self.match_to_reference(emb)
                if matched:
                    assignments[img_path] = matched
                else:
                    unknown_items.append((img_path, emb))
                continue

            # Multi-face image: try to match *every* face against references.
            matches = []
            for emb, _area in faces:
                m = self.match_to_reference(emb)
                if m is not None:
                    matches.append(m)

            unique_matches = set(matches)
            if len(unique_matches) == 1:
                assignments[img_path] = matches[0]
            elif len(unique_matches) > 1:
                multi_face_paths.append(img_path)
            else:
                # No reference matched any face.
                if self.reference_embeddings:
                    multi_face_paths.append(img_path)
                else:
                    unknown_items.append((img_path, faces[0][0]))

        # ── Pass 2: cluster unknowns ─────────────────────────────────────────
        cluster_assignments: Dict[Path, str] = {}

        if unknown_items:
            if len(unknown_items) >= self.cluster_min_samples:
                cluster_assignments = self._cluster_unknowns(unknown_items, out_path)
            else:
                # Too few images to cluster meaningfully — put in unknown/
                for path, _ in unknown_items:
                    cluster_assignments[path] = "unknown"

        # ── Pass 3: apply max_characters limit ───────────────────────────────
        all_assignments = {**assignments, **cluster_assignments}
        for path in no_face_paths:
            all_assignments[path] = "no_face"
        for path in multi_face_paths:
            all_assignments[path] = "multi_face"

        all_assignments = self._apply_max_characters(all_assignments, max_characters)

        # Optional quality-ranked trim: keep top-N sharpest per character.
        if max_per_character > 0:
            all_assignments = self.trim_per_character(
                all_assignments, max_per_character,
            )

        stats: Dict[str, int] = {}
        errors = 0

        for img_path, char_name in all_assignments.items():
            dest_dir = out_path / char_name
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / img_path.name

                # Resolve both sides so src==dest comparison is meaningful
                # across symlinks and relative paths.
                try:
                    same_file = (
                        img_path.resolve() == dest_file.resolve()
                    )
                except OSError:
                    same_file = False
                if same_file:
                    # Nothing to do — already at the destination. Still count
                    # it so the stats match user expectations.
                    stats[char_name] = stats.get(char_name, 0) + 1
                    continue

                # Handle name collisions
                if dest_file.exists():
                    stem, suffix = img_path.stem, img_path.suffix
                    counter = 1
                    while dest_file.exists():
                        dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                if copy:
                    shutil.copy2(str(img_path), str(dest_file))
                else:
                    # shutil.move handles cross-device fallback correctly;
                    # on same-device moves it's effectively an os.rename.
                    shutil.move(str(img_path), str(dest_file))
                stats[char_name] = stats.get(char_name, 0) + 1
            except (OSError, shutil.Error) as e:
                errors += 1
                logger.error(
                    "Failed to %s %s → %s: %s",
                    "copy" if copy else "move", img_path, dest_dir, e,
                )

        if errors:
            print(f"⚠️  {errors} file(s) failed to {'copy' if copy else 'move'} — see log")

        # ── Summary ──────────────────────────────────────────────────────────
        self._print_summary(stats, out_path, copy)

        # ── Manifest: write a machine-readable record of this run ───────────
        try:
            manifest = {
                "version": 2,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "duration_seconds": round(time.monotonic() - run_started, 2),
                "input_dir": str(input_path),
                "output_dir": str(out_path),
                "total_images": total,
                "stats": stats,
                "errors": errors,
                "settings": {
                    "model": self.model_name,
                    "similarity_threshold": self.similarity_threshold,
                    "match_margin": self.match_margin,
                    "cluster_eps": self.cluster_eps,
                    "cluster_min_samples": self.cluster_min_samples,
                    "max_characters": max_characters,
                    "copy": copy,
                    "recursive": recursive,
                    "num_workers": self.num_workers,
                    "cache_enabled": self._cache is not None,
                },
                "cache_stats": (
                    self._cache.stats() if self._cache is not None else None
                ),
                "references": {
                    name: int(arr.shape[0])
                    for name, arr in self.reference_embeddings.items()
                },
            }
            manifest_path = out_path / "_manifest.json"
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            logger.info("Manifest written: %s", manifest_path)
        except Exception as e:
            # Manifest failure is non-fatal — the sort itself already succeeded.
            logger.warning("Failed to write manifest: %s", e)

        return stats

    # ─── Fast directory walk ─────────────────────────────────────────────────

    def _scan_images(
        self,
        input_path: Path,
        out_path: Path,
        recursive: bool,
    ) -> List[Path]:
        """
        Fast scandir-based image walk. Skips anything inside ``out_path`` and
        any directory whose name begins with '_' (our convention for
        already-sorted output) — checks are performed at the *directory*
        level so we never descend into those subtrees at all, rather than
        filtering file-by-file.
        """
        try:
            out_resolved = out_path.resolve()
        except OSError:
            out_resolved = out_path

        def _should_skip_dir(dir_path: Path) -> bool:
            # Skip any sub-directory whose basename starts with '_' (that's
            # how previous sort runs mark their output).
            if dir_path.name.startswith('_'):
                return True
            # Skip the actual output directory (even if its name doesn't
            # start with '_') to prevent recursive self-ingestion.
            try:
                if dir_path.resolve() == out_resolved:
                    return True
            except OSError:
                pass
            return False

        results: List[Path] = []
        supported = self.SUPPORTED_EXTENSIONS

        # Iterative DFS with an explicit stack — avoids Python's recursion
        # limit on very deep directory trees and keeps the per-frame cost
        # as low as a scandir iterator.
        stack: List[Path] = [input_path]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        try:
                            if entry.is_dir(follow_symlinks=False):
                                if recursive and not _should_skip_dir(Path(entry.path)):
                                    stack.append(Path(entry.path))
                            elif entry.is_file(follow_symlinks=False):
                                name = entry.name
                                dot = name.rfind('.')
                                if dot == -1:
                                    continue
                                if name[dot:].lower() not in supported:
                                    continue
                                results.append(Path(entry.path))
                        except OSError as e:
                            logger.debug("scandir entry error for %s: %s", entry.path, e)
            except OSError as e:
                logger.warning("Cannot read directory %s: %s", d, e)

        # Deterministic ordering so progress/tests are reproducible.
        results.sort()
        return results

    # ─── Clustering helpers ───────────────────────────────────────────────────

    def _cluster_unknowns(
        self,
        items: List[Tuple[Path, np.ndarray]],
        out_path: Optional[Path] = None,
    ) -> Dict[Path, str]:
        """
        DBSCAN clustering on face embeddings.
        Returns {path: folder_name}.
        """
        DBSCAN = _get_dbscan()

        embeddings = np.stack([emb for _, emb in items])  # (N, 512)

        # Use cosine metric via precomputed distance matrix
        # cosine_distance = 1 - dot_product (for normalised vectors)
        dot = embeddings @ embeddings.T
        dot = np.clip(dot, -1.0, 1.0)
        dist_matrix = 1.0 - dot

        clustering = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            metric='precomputed',
        ).fit(dist_matrix.astype(np.float64))

        labels = clustering.labels_  # -1 = noise

        # Avoid colliding with reference character names *and* any
        # character_XX folders that already exist in the output directory
        # (e.g. from a previous run).
        existing = set(self.reference_embeddings.keys())
        if out_path is not None and out_path.exists():
            for child in out_path.iterdir():
                if child.is_dir():
                    existing.add(child.name)

        cluster_idx = 1
        label_to_name: Dict[int, str] = {}

        for label in sorted(set(labels)):
            if label == -1:
                label_to_name[-1] = "unknown"
            else:
                name = f"character_{cluster_idx:02d}"
                while name in existing:
                    cluster_idx += 1
                    name = f"character_{cluster_idx:02d}"
                label_to_name[label] = name
                existing.add(name)
                cluster_idx += 1

        result: Dict[Path, str] = {}
        for (path, _), label in zip(items, labels):
            result[path] = label_to_name[label]

        return result

    # ─── max_characters limiter ───────────────────────────────────────────────

    def _apply_max_characters(
        self,
        assignments: Dict[Path, str],
        max_characters: int,
    ) -> Dict[Path, str]:
        """
        Enforce a maximum number of character folders.

        Counts images per character name (excluding system folders).
        Keeps the top `max_characters` largest groups; everything else
        is reassigned to "other/".
        """
        max_characters = max(1, min(max_characters, 6))

        # Count images per character (ignore system folders)
        char_counts: Dict[str, int] = {}
        for name in assignments.values():
            if name not in self._SYSTEM_FOLDERS:
                char_counts[name] = char_counts.get(name, 0) + 1

        if len(char_counts) <= max_characters:
            return assignments  # already within limit, nothing to do

        # Keep top N by count; ties resolved by name (alphabetical)
        top_chars = set(
            name for name, _ in sorted(
                char_counts.items(),
                key=lambda x: (-x[1], x[0])
            )[:max_characters]
        )

        overflow = len(char_counts) - max_characters
        print(f"  ⚠️  max_characters={max_characters}: merging {overflow} smaller group(s) → other/")

        result = {}
        for path, name in assignments.items():
            if name not in self._SYSTEM_FOLDERS and name not in top_chars:
                result[path] = "other"
            else:
                result[path] = name
        return result

    # ─── Quality-ranked trim ────────────────────────────────────────────────

    @staticmethod
    def _sharpness(path: Path) -> float:
        """Laplacian variance — higher = sharper. Returns 0.0 on failure."""
        img = _imread_unicode(path)
        if img is None:
            return 0.0
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def trim_per_character(
        self,
        assignments: Dict[Path, str],
        max_per_character: int,
    ) -> Dict[Path, str]:
        """
        Within each character group, keep only the ``max_per_character``
        sharpest images (by Laplacian variance). Excess images are
        re-assigned to ``trimmed/``.

        System folders (no_face, multi_face, unknown, other) are not
        trimmed.

        Args:
            assignments: {path → character_name} dict.
            max_per_character: Keep at most this many per character.
                               0 or negative → no trimming.

        Returns:
            Updated assignments dict.
        """
        if max_per_character <= 0:
            return assignments

        # Group paths by character, skipping system folders.
        from collections import defaultdict
        groups: Dict[str, List[Path]] = defaultdict(list)
        for path, name in assignments.items():
            groups[name].append(path)

        trimmed_count = 0
        result = dict(assignments)
        for name, paths in groups.items():
            if name in self._SYSTEM_FOLDERS:
                continue
            if len(paths) <= max_per_character:
                continue
            # Rank by sharpness descending
            scored = sorted(
                ((p, self._sharpness(p)) for p in paths),
                key=lambda x: x[1],
                reverse=True,
            )
            keep = {p for p, _ in scored[:max_per_character]}
            for p in paths:
                if p not in keep:
                    result[p] = "trimmed"
                    trimmed_count += 1

        if trimmed_count:
            print(f"  ✂  Trimmed {trimmed_count} excess image(s) → trimmed/")

        return result

    # ─── Utility ──────────────────────────────────────────────────────────────

    def _print_summary(self, stats: Dict[str, int], out_path: Path, copy: bool):
        action_word = "Copied" if copy else "Moved"
        total = sum(stats.values())
        print(f"\n{'─'*50}")
        print(f"Character Sort Complete — {action_word} {total} files")
        print(f"Output: {out_path}")
        print(f"{'─'*50}")
        for name, count in sorted(stats.items()):
            icon = "👤" if not name.startswith(("no_face", "multi_face", "unknown")) else "📁"
            print(f"  {icon} {name:25s} {count:4d} image(s)")
        print(f"{'─'*50}")

    # ─── Convenience: sort a single image ────────────────────────────────────

    def identify_image(self, image_path: str) -> str:
        """
        Identify the character in a single image.
        Returns character name, 'unknown', 'no_face', or 'multi_face'.
        """
        self._load_model()
        img = _imread_unicode(image_path)
        if img is None:
            return "no_face"

        emb = self.get_largest_face_embedding(img)
        if emb is None:
            return "no_face"

        matched = self.match_to_reference(emb)
        return matched if matched else "unknown"
