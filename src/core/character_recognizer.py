"""
Character Recognizer - InsightFace based face recognition with hybrid matching
Supports:
  - Reference-based matching (supervised): match faces to known characters
  - Auto-clustering (unsupervised): group unknown faces automatically
  - Hybrid mode: reference match first, cluster the rest
"""

import os
import cv2
import shutil
import logging
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

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
        """
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.similarity_threshold = similarity_threshold
        self.match_margin = match_margin
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.progress_callback = progress_callback

        # {character_name: np.ndarray of shape (N, 512)}
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        # Mean embedding per character (kept for backwards compatibility;
        # matching now prefers the full per-sample distance, see
        # ``match_to_reference``).
        self.reference_means: Dict[str, np.ndarray] = {}

        self._app = None  # InsightFace FaceAnalysis (lazy)
        self._app_lock = threading.Lock()  # guard lazy model init across threads

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
            app = FaceAnalysis(
                name=self.model_name,
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

    def _detect_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, float]]:
        """
        Run InsightFace once and return [(normalised_embedding, bbox_area), ...]
        sorted by bbox area descending. This is the single entry point used by
        both ``get_embeddings`` and ``get_largest_face_embedding`` so we never
        run the detection pipeline twice on the same image.
        """
        self._load_model()
        faces = self._app.get(image)  # BGR expected (OpenCV default)
        results: List[Tuple[np.ndarray, float]] = []
        for face in faces:
            if face.embedding is None:
                continue
            bbox = face.bbox
            area = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            results.append((self._normalise(face.embedding), area))
        # Sort descending by area so results[0] is always the largest face
        results.sort(key=lambda x: x[1], reverse=True)
        return results

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
        counts = {}

        char_dirs = [d for d in sorted(ref_path.iterdir()) if d.is_dir()]
        if not char_dirs:
            logger.warning("No character sub-folders found in %s", ref_path)
            return {}

        print(f"Loading references from: {ref_path}")
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

            for img_path in image_files:
                img = _imread_unicode(img_path)
                if img is None:
                    unreadable += 1
                    logger.warning("Could not read reference image: %s", img_path)
                    continue

                faces = self._detect_faces(img)
                if not faces:
                    no_face += 1
                    logger.warning("No face detected in reference image: %s", img_path)
                    continue

                # Use only the largest face from each reference. Warn the user
                # if extra faces were silently discarded — they should crop
                # their references so each image contains only the target
                # character.
                if len(faces) > 1:
                    multi_face_refs += 1
                    logger.info(
                        "Reference %s contains %d faces; using only the largest. "
                        "Consider cropping to a single face for better accuracy.",
                        img_path, len(faces),
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

        return counts

    # ─── Matching ─────────────────────────────────────────────────────────────

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two normalised vectors. Range [0, 2]."""
        return float(1.0 - np.dot(a, b))

    def match_to_reference(self, embedding: np.ndarray) -> Optional[str]:
        """
        Match a query embedding to the closest reference character.

        For each character the distance is the *minimum* cosine distance to
        any of that character's reference embeddings — this preserves recall
        when a character is represented by multiple angles/styles (taking a
        mean embedding blurs them together).

        Returns the character name only when:
          - the best distance is within ``similarity_threshold``, AND
          - the gap to the second-best character is at least ``match_margin``
            (otherwise the result is ambiguous and we return None so the
            image is handed off to the clustering stage).
        """
        if not self.reference_embeddings:
            return None

        # Compute per-character min distance using vectorised dot product.
        ranked: List[Tuple[str, float]] = []
        for char_name, emb_array in self.reference_embeddings.items():
            # emb_array: (N, 512); embedding: (512,)
            sims = emb_array @ embedding  # (N,)
            # Best (smallest) distance for this character
            best = float(1.0 - np.max(sims))
            ranked.append((char_name, best))

        ranked.sort(key=lambda x: x[1])
        best_name, best_dist = ranked[0]

        if best_dist > self.similarity_threshold:
            return None

        # Ambiguity check: if the runner-up is almost as close, reject the
        # match rather than guessing.
        if len(ranked) >= 2:
            _, runner_up_dist = ranked[1]
            if (runner_up_dist - best_dist) < self.match_margin:
                logger.debug(
                    "Ambiguous match: %s@%.3f vs %s@%.3f (margin=%.3f)",
                    best_name, best_dist, ranked[1][0], runner_up_dist,
                    self.match_margin,
                )
                return None

        return best_name

    # ─── Main sorting pipeline ────────────────────────────────────────────────

    # System folder names that are never counted toward the character limit
    _SYSTEM_FOLDERS = {"no_face", "multi_face", "unknown", "other"}

    def sort_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        copy: bool = False,
        recursive: bool = False,
        max_characters: int = 6,
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

        out_path = (
            Path(output_dir).resolve() if output_dir
            else input_path / "_sorted"
        )
        out_path.mkdir(parents=True, exist_ok=True)

        # Collect image files. We must be careful NOT to pick up files that
        # live inside the output directory (common when running twice) or
        # inside any previously-sorted directory whose name starts with '_'.
        # The original implementation only inspected the immediate parent
        # which broke with recursive=True.
        def _is_under(path: Path, base: Path) -> bool:
            try:
                path.resolve().relative_to(base)
                return True
            except ValueError:
                return False

        def _has_sorted_component(path: Path) -> bool:
            try:
                rel = path.resolve().relative_to(input_path)
            except ValueError:
                return False
            return any(part.startswith('_') for part in rel.parts)

        pattern = "**/*" if recursive else "*"
        image_files: List[Path] = []
        for f in input_path.glob(pattern):
            if not f.is_file():
                continue
            if f.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
                continue
            # Skip anything inside the output directory (prevents re-processing)
            if _is_under(f, out_path):
                continue
            # Skip any path containing a "_*" component anywhere in the chain
            if _has_sorted_component(f):
                continue
            image_files.append(f)

        if not image_files:
            print(f"No image files found in {input_dir}")
            return {}

        self._load_model()

        total = len(image_files)
        print(f"\nProcessing {total} images...")
        print(f"References loaded: {list(self.reference_embeddings.keys()) or 'None (auto-cluster only)'}")
        print(f"Similarity threshold: {self.similarity_threshold}  margin: {self.match_margin}")

        # ── Pass 1: extract embeddings & do reference matching ──────────────
        unknown_items: List[Tuple[Path, np.ndarray]] = []   # (path, embedding)
        no_face_paths: List[Path] = []
        multi_face_paths: List[Path] = []
        assignments: Dict[Path, str] = {}  # path → character name

        for idx, img_path in enumerate(image_files, 1):
            if self.progress_callback:
                self.progress_callback(idx, total, f"Scanning {img_path.name}")

            img = _imread_unicode(img_path)
            if img is None:
                # File exists but could not be decoded — this is an error,
                # not "no face". Log it and skip (don't pollute no_face/).
                logger.warning("Failed to decode image: %s", img_path)
                continue

            faces = self._detect_faces(img)  # single inference

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
            # If exactly one unambiguously matches, assign to that character.
            # If several different characters match (or nothing matches at all
            # but we still have multiple faces), route to multi_face/.
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
                # No reference matched any face. In auto-cluster mode we can
                # still try clustering using the largest face. In pure
                # reference mode with no matches at all, this is multi_face.
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

        stats: Dict[str, int] = {}
        errors = 0
        action = shutil.copy2 if copy else shutil.move

        for img_path, char_name in all_assignments.items():
            dest_dir = out_path / char_name
            try:
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_file = dest_dir / img_path.name

                # Handle name collisions
                if dest_file.exists():
                    stem, suffix = img_path.stem, img_path.suffix
                    counter = 1
                    while dest_file.exists():
                        dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                        counter += 1

                action(str(img_path), str(dest_file))
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
        return stats

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
