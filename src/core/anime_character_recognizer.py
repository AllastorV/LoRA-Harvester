"""
Anime Character Recognizer
--------------------------
Detection : lbpcascade_animeface (Haar cascade, ~1.5 MB XML)
Embedding : ResNet-18 ImageNet features  (512-d, L2-normalized)
Clustering: DBSCAN  — same algorithm as the real-face recognizer

Public API is identical to CharacterRecognizer so CharacterSortThread
can use either class without changes.
"""

import os
import cv2
import shutil
import sqlite3
import logging
import threading
import urllib.request
import numpy as np
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

# ─── cascade ─────────────────────────────────────────────────────────────────

_CASCADE_URL  = (
    "https://raw.githubusercontent.com/nagadomi/"
    "lbpcascade_animeface/master/lbpcascade_animeface.xml"
)
_CASCADE_PATH = (
    Path(__file__).parent.parent.parent
    / "models" / "anime" / "lbpcascade_animeface.xml"
)

# ─── helpers ──────────────────────────────────────────────────────────────────

def _imread_unicode(path) -> Optional[np.ndarray]:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None
    except Exception:
        return None


def _cosine_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance in [0, 2].  0 = identical, 2 = opposite."""
    return float(1.0 - float(np.dot(a, b)))


# ─── SQLite cache (same schema as EmbeddingCache) ────────────────────────────

class _AnimeCache:
    _MODEL_KEY = "anime_resnet18"

    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                path     TEXT,
                model    TEXT,
                mtime_ns INTEGER,
                size     INTEGER,
                n_faces  INTEGER,
                data     BLOB,
                PRIMARY KEY (path, model)
            )
        """)
        self._conn.commit()
        self._lock = threading.Lock()
        self._dirty = 0

    def get(self, path: Path) -> Optional[np.ndarray]:
        """Return (N, 512) float32 or None on cache miss."""
        try:
            stat = os.stat(str(path))
            with self._lock:
                row = self._conn.execute(
                    "SELECT mtime_ns, size, n_faces, data FROM embeddings "
                    "WHERE path=? AND model=?",
                    (str(path), self._MODEL_KEY),
                ).fetchone()
            if row is None:
                return None
            mtime_ns, size, n_faces, blob = row
            if mtime_ns != stat.st_mtime_ns or size != stat.st_size:
                return None  # stale
            if n_faces == 0:
                return np.empty((0, 512), dtype=np.float32)
            arr = np.frombuffer(blob, dtype=np.float32).reshape(n_faces, 512).copy()
            return arr
        except Exception:
            return None

    def put(self, path: Path, embeddings: np.ndarray):
        """Store (N, 512) array. Call with empty array for 0-face result."""
        try:
            stat = os.stat(str(path))
            blob = embeddings.astype(np.float32).tobytes()
            with self._lock:
                self._conn.execute(
                    "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?,?,?)",
                    (str(path), self._MODEL_KEY,
                     stat.st_mtime_ns, stat.st_size,
                     len(embeddings), blob),
                )
                self._dirty += 1
                # Commit every 16 writes — small enough batches so a crash
                # loses at most 16 embeddings, large enough to amortize I/O.
                if self._dirty >= 16:
                    self._conn.commit()
                    self._dirty = 0
        except Exception:
            pass

    def close(self):
        try:
            with self._lock:
                self._conn.commit()
                self._conn.close()
        except Exception:
            pass


# ─── main class ───────────────────────────────────────────────────────────────

class AnimeCharacterRecognizer:
    """
    Anime-optimised character recognizer.

    Parameters mirror CharacterRecognizer so both can be used
    interchangeably from CharacterSortThread.
    """

    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

    def __init__(
        self,
        use_gpu: bool = True,
        cluster_eps: float = 0.55,
        cluster_min_samples: int = 2,
        similarity_threshold: float = 0.55,   # cosine dist threshold for reference match
        match_margin: float = 0.05,
        progress_callback=None,
        num_workers: int = 4,
        use_cache: bool = True,
        cache_path: Optional[str] = None,
        reference_dir: Optional[str] = None,   # ignored at init, used in load_references
        # Ignored kwargs forwarded from the thread (model_name, etc.)
        **_,
    ):
        self.use_gpu = use_gpu
        self.device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.similarity_threshold = similarity_threshold
        self.match_margin = match_margin
        self.progress_callback = progress_callback
        self.num_workers = num_workers

        # reference embeddings: {char_name: list[np.ndarray(512,)]}
        self.reference_embeddings: Dict[str, List[np.ndarray]] = {}

        # Models — lazy-loaded on first use to keep __init__ fast
        self._cascade: Optional[cv2.CascadeClassifier] = None
        self._embedder = None
        self._transform = None

        # Cache
        self._cache: Optional[_AnimeCache] = None
        self._use_cache = use_cache
        self._cache_path = Path(cache_path) if cache_path else None

    # ── model loading ─────────────────────────────────────────────────────────

    def _ensure_models(self):
        if self._cascade is not None:
            return
        self._cascade  = self._load_cascade()
        self._embedder, self._transform = self._load_embedder()

    @staticmethod
    def _load_cascade() -> cv2.CascadeClassifier:
        if not _CASCADE_PATH.exists():
            print(f"Downloading lbpcascade_animeface.xml …")
            _CASCADE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = _CASCADE_PATH.with_suffix(".tmp")
            try:
                urllib.request.urlretrieve(_CASCADE_URL, str(tmp))
                tmp.replace(_CASCADE_PATH)   # atomic rename — no partial file left on failure
                print("  ✅  cascade downloaded")
            except Exception as e:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Could not download anime face cascade: {e}\n"
                    f"Place lbpcascade_animeface.xml in {_CASCADE_PATH}"
                )
        cascade = cv2.CascadeClassifier(str(_CASCADE_PATH))
        if cascade.empty():
            # File may be corrupt — delete so next run re-downloads
            _CASCADE_PATH.unlink(missing_ok=True)
            raise RuntimeError(
                f"Failed to load cascade from {_CASCADE_PATH} (file deleted, will re-download next run)"
            )
        return cascade

    def _load_embedder(self):
        try:
            import torchvision.models as M
            import torchvision.transforms as T
        except ImportError:
            raise RuntimeError(
                "torchvision is required for anime mode.\n"
                "Install: pip install torchvision"
            )
        model = M.resnet18(weights="IMAGENET1K_V1")
        model.fc = torch.nn.Identity()   # drop classifier → 512-d pool output
        model.eval()
        if self.device == "cuda":
            model = model.cuda()

        transform = T.Compose([
            T.ToPILImage(),
            T.Resize((112, 112)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
        return model, transform

    def _open_cache(self, base_path: Path):
        if not self._use_cache:
            return
        db = self._cache_path or (base_path / ".anime_embed_cache.db")
        if self._cache is None:
            self._cache = _AnimeCache(db)

    # ── face detection & embedding ────────────────────────────────────────────

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Return list of (x, y, w, h) bounding boxes."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    def _embed_crop(self, image: np.ndarray, bbox: Tuple[int,int,int,int]) -> Optional[np.ndarray]:
        x, y, w, h = bbox
        m = int(max(w, h) * 0.15)
        x1 = max(0, x - m);  y1 = max(0, y - m)
        x2 = min(image.shape[1], x + w + m)
        y2 = min(image.shape[0], y + h + m)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None
        try:
            # cv2 reads BGR; ToPILImage / ResNet-18 expect RGB — convert first
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                t = self._transform(crop_rgb).unsqueeze(0)
                if self.device == "cuda":
                    t = t.cuda()
                emb = self._embedder(t).squeeze().cpu().float().numpy()
        except Exception:
            return None
        norm = np.linalg.norm(emb)
        return (emb / norm) if norm > 1e-8 else emb

    def _get_embeddings(self, img_path: Path) -> np.ndarray:
        """Return (N, 512) float32 array.  N=0 means no face detected."""
        if self._cache is not None:
            cached = self._cache.get(img_path)
            if cached is not None:
                return cached

        img = _imread_unicode(img_path)
        if img is None:
            return np.empty((0, 512), dtype=np.float32)

        faces = self._detect_faces(img)
        embs = []
        for bbox in faces:
            e = self._embed_crop(img, bbox)
            if e is not None:
                embs.append(e)

        result = (np.array(embs, dtype=np.float32)
                  if embs else np.empty((0, 512), dtype=np.float32))

        if self._cache is not None:
            self._cache.put(img_path, result)

        return result

    # ── reference matching ────────────────────────────────────────────────────

    def load_references(self, reference_dir: Optional[str] = None) -> Dict[str, int]:
        """Load reference images and build per-character embedding lists."""
        if reference_dir is None:
            return {}
        ref_path = Path(reference_dir)
        if not ref_path.exists():
            raise FileNotFoundError(f"Reference dir not found: {ref_path}")

        self._ensure_models()
        self._open_cache(ref_path)
        self.reference_embeddings.clear()
        counts: Dict[str, int] = {}

        for char_dir in sorted(ref_path.iterdir()):
            if not char_dir.is_dir():
                continue
            name = char_dir.name
            embs = []
            for img_path in sorted(char_dir.iterdir()):
                if img_path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                    continue
                arr = self._get_embeddings(img_path)
                if len(arr) > 0:
                    embs.append(arr[0])
            if embs:
                self.reference_embeddings[name] = embs
                counts[name] = len(embs)

        return counts

    def _match_to_reference(self, emb: np.ndarray) -> Optional[str]:
        """Return best-match character name, or None if below threshold."""
        if not self.reference_embeddings:
            return None
        best_name, best_dist = None, float("inf")
        second_dist = float("inf")

        for name, ref_list in self.reference_embeddings.items():
            dist = min(_cosine_dist(emb, r) for r in ref_list)
            if dist < best_dist:
                second_dist = best_dist
                best_dist, best_name = dist, name
            elif dist < second_dist:
                second_dist = dist

        if best_dist > self.similarity_threshold:
            return None
        if (second_dist - best_dist) < self.match_margin:
            return None   # ambiguous
        return best_name

    # ── image scanner ─────────────────────────────────────────────────────────

    def _scan_images(self, input_path: Path, out_path: Path, recursive: bool) -> List[Path]:
        files: List[Path] = []
        q = deque([input_path])
        while q:
            d = q.popleft()
            try:
                with os.scandir(str(d)) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            ep = Path(entry.path)
                            if recursive and ep != out_path:
                                q.append(ep)
                        elif entry.is_file(follow_symlinks=False):
                            if Path(entry.name).suffix.lower() in self.SUPPORTED_EXTENSIONS:
                                files.append(Path(entry.path))
            except PermissionError:
                pass
        return sorted(files)

    # ── file move / copy ──────────────────────────────────────────────────────

    @staticmethod
    def _safe_copy_move(src: Path, dst_dir: Path, copy: bool):
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / src.name
        if dst.exists() and dst != src:
            # st_ino is unreliable on Windows (often 0) — use a counter instead
            stem, suffix = src.stem, src.suffix
            counter = 1
            while dst.exists():
                dst = dst_dir / f"{stem}_{counter}{suffix}"
                counter += 1
        if copy:
            shutil.copy2(str(src), str(dst))
        else:
            shutil.move(str(src), str(dst))

    # ── clustering ────────────────────────────────────────────────────────────

    def _cluster(self, embeddings: np.ndarray) -> np.ndarray:
        """DBSCAN on cosine distances. Returns label array (-1 = noise)."""
        try:
            from sklearn.cluster import DBSCAN
            from sklearn.metrics.pairwise import cosine_distances
        except ImportError:
            raise RuntimeError(
                "scikit-learn is required for clustering.\n"
                "Install: pip install scikit-learn"
            )
        if len(embeddings) < self.cluster_min_samples:
            return np.full(len(embeddings), -1, dtype=int)
        dist_matrix = cosine_distances(embeddings)
        np.clip(dist_matrix, 0, 2, out=dist_matrix)
        labels = DBSCAN(
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
            metric="precomputed",
        ).fit_predict(dist_matrix)
        return labels

    # ── public API ────────────────────────────────────────────────────────────

    def sort_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        copy: bool = False,
        recursive: bool = True,
        max_characters: int = 6,
        max_per_character: int = 0,
    ) -> Dict[str, int]:
        """
        Sort images by anime character.

        Returns dict {category_name: count}.
        Category names: reference names, 'character_01' …, 'no_face',
        'multi_face', 'other'.
        """
        self._ensure_models()

        input_path = Path(input_dir)
        out_path   = Path(output_dir) if output_dir else input_path / "_sorted"
        if out_path == input_path:
            out_path = input_path / "_sorted"
        out_path.mkdir(parents=True, exist_ok=True)

        self._open_cache(input_path)

        image_files = self._scan_images(input_path, out_path, recursive)
        total = len(image_files)

        if total == 0:
            print(f"No images found in {input_dir}")
            return {}

        print(f"\nProcessing {total} images (anime mode)…")
        print(f"Detection: lbpcascade_animeface  |  Embedding: ResNet-18")
        print(f"Threshold: {self.similarity_threshold}  eps: {self.cluster_eps}")

        # ── Pass 1: extract embeddings ────────────────────────────────────────
        no_face_paths:    List[Path] = []
        single_face:      List[Tuple[Path, np.ndarray]] = []   # (path, emb)
        multi_face_paths: List[Path] = []

        for i, img_path in enumerate(image_files, 1):
            if self.progress_callback:
                self.progress_callback(i, total, img_path.name)

            embs = self._get_embeddings(img_path)
            n = len(embs)

            if n == 0:
                no_face_paths.append(img_path)
            elif n == 1:
                single_face.append((img_path, embs[0]))
            else:
                # multiple faces — try to match each to a reference
                matched_any = False
                for emb in embs:
                    m = self._match_to_reference(emb)
                    if m:
                        single_face.append((img_path, emb))
                        matched_any = True
                        break
                if not matched_any:
                    multi_face_paths.append(img_path)

        stats: Dict[str, int] = {}

        # ── no_face ───────────────────────────────────────────────────────────
        nf_dir = out_path / "no_face"
        for p in no_face_paths:
            self._safe_copy_move(p, nf_dir, copy)
        if no_face_paths:
            stats["no_face"] = len(no_face_paths)

        # ── multi_face ────────────────────────────────────────────────────────
        mf_dir = out_path / "multi_face"
        for p in multi_face_paths:
            self._safe_copy_move(p, mf_dir, copy)
        if multi_face_paths:
            stats["multi_face"] = len(multi_face_paths)

        if not single_face:
            self._print_summary(out_path, copy, stats)
            return stats

        # ── Pass 2: reference matching ────────────────────────────────────────
        unknown: List[Tuple[Path, np.ndarray]] = []

        for path, emb in single_face:
            matched = self._match_to_reference(emb)
            if matched:
                self._safe_copy_move(path, out_path / matched, copy)
                stats[matched] = stats.get(matched, 0) + 1
            else:
                unknown.append((path, emb))

        # ── Pass 3: DBSCAN clustering of unmatched single-face images ─────────
        if unknown:
            paths_u = [p for p, _ in unknown]
            embs_u  = np.array([e for _, e in unknown], dtype=np.float32)
            labels  = self._cluster(embs_u)

            unique_labels = sorted(set(labels) - {-1})
            # Limit to max_characters (merge excess into 'other')
            char_labels = unique_labels[:max_characters]
            other_labels = set(unique_labels[max_characters:])

            label_to_name: Dict[int, str] = {}
            char_idx = 1
            for lbl in unique_labels:
                if lbl in other_labels:
                    label_to_name[lbl] = "other"
                else:
                    label_to_name[lbl] = f"character_{char_idx:02d}"
                    char_idx += 1

            for path, lbl in zip(paths_u, labels):
                if lbl == -1:
                    cat = "other"
                else:
                    cat = label_to_name[lbl]

                # max_per_character cap
                if max_per_character > 0 and stats.get(cat, 0) >= max_per_character:
                    cat = "other"

                self._safe_copy_move(path, out_path / cat, copy)
                stats[cat] = stats.get(cat, 0) + 1

        self._print_summary(out_path, copy, stats)
        return stats

    def close_cache(self):
        if self._cache is not None:
            self._cache.close()
            self._cache = None

    @staticmethod
    def _print_summary(out_path: Path, copy: bool, stats: Dict[str, int]):
        total = sum(stats.values())
        action = "Copied" if copy else "Moved"
        print(f"\n{'─'*50}")
        print(f"Character Sort Complete — {action} {total} files")
        print(f"Output: {out_path}")
        print(f"{'─'*50}")
        for k, v in sorted(stats.items()):
            icon = "👤" if k.startswith("character_") else "📁"
            print(f"  {icon} {k:<30} {v} image(s)")
        print(f"{'─'*50}")
