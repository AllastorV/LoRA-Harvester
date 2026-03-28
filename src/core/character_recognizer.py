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
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Lazy imports to avoid hard crash if not installed
_insightface = None
_sklearn = None


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
            cluster_eps: DBSCAN eps parameter for clustering unknowns.
            cluster_min_samples: DBSCAN min_samples parameter.
            use_gpu: Use GPU if available.
            model_name: InsightFace model pack ('buffalo_l' recommended).
            progress_callback: Optional callable(current, total, message).
        """
        self.reference_dir = Path(reference_dir) if reference_dir else None
        self.similarity_threshold = similarity_threshold
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.use_gpu = use_gpu
        self.model_name = model_name
        self.progress_callback = progress_callback

        # {character_name: np.ndarray of shape (N, 512)}
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        # Mean embedding per character for fast matching
        self.reference_means: Dict[str, np.ndarray] = {}

        self._app = None  # InsightFace FaceAnalysis (lazy)

    # ─── InsightFace initialization ───────────────────────────────────────────

    def _load_model(self):
        if self._app is not None:
            return
        FaceAnalysis = _get_insightface()
        ctx_id = 0 if self.use_gpu else -1
        self._app = FaceAnalysis(name=self.model_name, allowed_modules=['detection', 'recognition'])
        self._app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        logger.info("InsightFace model loaded (ctx_id=%d)", ctx_id)

    # ─── Embedding extraction ──────────────────────────────────────────────────

    def get_embeddings(self, image: np.ndarray) -> List[np.ndarray]:
        """
        Detect all faces in an image and return their L2-normalised embeddings.
        Returns empty list if no face detected.
        """
        self._load_model()
        # InsightFace expects BGR (OpenCV default)
        faces = self._app.get(image)
        embeddings = []
        for face in faces:
            if face.embedding is not None:
                emb = face.embedding.astype(np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-6)
                embeddings.append(emb)
        return embeddings

    def get_largest_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Return embedding of the largest face only (most likely the main subject)."""
        self._load_model()
        faces = self._app.get(image)
        if not faces:
            return None
        # Pick largest face by bounding-box area
        largest = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        if largest.embedding is None:
            return None
        emb = largest.embedding.astype(np.float32)
        emb = emb / (np.linalg.norm(emb) + 1e-6)
        return emb

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
            embeddings = []

            image_files = [
                f for f in sorted(char_dir.iterdir())
                if f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]

            for img_path in image_files:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                emb = self.get_largest_face_embedding(img)
                if emb is not None:
                    embeddings.append(emb)

            if embeddings:
                emb_array = np.stack(embeddings)  # (N, 512)
                self.reference_embeddings[char_name] = emb_array
                self.reference_means[char_name] = emb_array.mean(axis=0)
                self.reference_means[char_name] /= (np.linalg.norm(self.reference_means[char_name]) + 1e-6)
                counts[char_name] = len(embeddings)
                print(f"  {char_name}: {len(embeddings)} reference(s) loaded")
            else:
                print(f"  {char_name}: No faces detected in references - skipping")

        return counts

    # ─── Matching ─────────────────────────────────────────────────────────────

    def cosine_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine distance between two normalised vectors. Range [0, 2]."""
        return float(1.0 - np.dot(a, b))

    def match_to_reference(self, embedding: np.ndarray) -> Optional[str]:
        """
        Match a query embedding to the closest reference character.
        Returns character name if within threshold, else None.
        """
        if not self.reference_means:
            return None

        best_name = None
        best_dist = float('inf')

        for char_name, mean_emb in self.reference_means.items():
            dist = self.cosine_distance(embedding, mean_emb)
            if dist < best_dist:
                best_dist = dist
                best_name = char_name

        if best_dist <= self.similarity_threshold:
            return best_name
        return None

    # ─── Main sorting pipeline ────────────────────────────────────────────────

    # System folder names that are never counted toward the character limit
    _SYSTEM_FOLDERS = {"no_face", "multi_face", "unknown", "other"}

    def sort_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        copy: bool = False,
        recursive: bool = False,
        max_characters: int = 1,
    ) -> Dict[str, int]:
        """
        Sort images in input_dir into character sub-folders.

        Mode:
          - If references loaded → tries reference matching first
          - Unmatched faces → auto-clustered into character_01/, character_02/ ...
          - No faces detected → no_face/
          - Multiple faces with no clear match → multi_face/
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
        input_path = Path(input_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        out_path = Path(output_dir) if output_dir else input_path / "_sorted"
        out_path.mkdir(parents=True, exist_ok=True)

        # Collect image files
        pattern = "**/*" if recursive else "*"
        image_files = [
            f for f in input_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.SUPPORTED_EXTENSIONS
            and not f.parent.name.startswith('_')  # skip already-sorted folders
        ]

        if not image_files:
            print(f"No image files found in {input_dir}")
            return {}

        self._load_model()

        total = len(image_files)
        print(f"\nProcessing {total} images...")
        print(f"References loaded: {list(self.reference_embeddings.keys()) or 'None (auto-cluster only)'}")
        print(f"Similarity threshold: {self.similarity_threshold}")

        # ── Pass 1: extract embeddings & do reference matching ──────────────
        unknown_items: List[Tuple[Path, np.ndarray]] = []   # (path, embedding)
        no_face_paths: List[Path] = []
        multi_face_paths: List[Path] = []
        assignments: Dict[Path, str] = {}  # path → character name

        for idx, img_path in enumerate(image_files, 1):
            if self.progress_callback:
                self.progress_callback(idx, total, f"Scanning {img_path.name}")

            img = cv2.imread(str(img_path))
            if img is None:
                no_face_paths.append(img_path)
                continue

            embeddings = self.get_embeddings(img)

            if not embeddings:
                no_face_paths.append(img_path)
                continue

            if len(embeddings) > 1:
                # Multiple faces: try largest-face only
                main_emb = self.get_largest_face_embedding(img)
                if main_emb is None:
                    multi_face_paths.append(img_path)
                    continue
                embeddings = [main_emb]

            emb = embeddings[0]

            # Try reference match
            matched = self.match_to_reference(emb)
            if matched:
                assignments[img_path] = matched
            else:
                unknown_items.append((img_path, emb))

        # ── Pass 2: cluster unknowns ─────────────────────────────────────────
        cluster_assignments: Dict[Path, str] = {}

        if unknown_items:
            if len(unknown_items) >= self.cluster_min_samples:
                cluster_assignments = self._cluster_unknowns(unknown_items)
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
        action = shutil.copy2 if copy else shutil.move

        for img_path, char_name in all_assignments.items():
            dest_dir = out_path / char_name
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

        # ── Summary ──────────────────────────────────────────────────────────
        self._print_summary(stats, out_path, copy)
        return stats

    # ─── Clustering helpers ───────────────────────────────────────────────────

    def _cluster_unknowns(
        self,
        items: List[Tuple[Path, np.ndarray]],
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

        # Find next free cluster index (don't overlap with reference names)
        existing = set(self.reference_embeddings.keys())
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
        img = cv2.imread(image_path)
        if img is None:
            return "no_face"

        emb = self.get_largest_face_embedding(img)
        if emb is None:
            return "no_face"

        matched = self.match_to_reference(emb)
        return matched if matched else "unknown"
