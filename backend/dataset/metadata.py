import logging
import re
from pathlib import Path
import pandas as pd

# Set up logging for the metadata manager
logger = logging.getLogger(__name__)


class MetadataManager:
    """Manages metadata for the INCLUDE dataset.

    Serves as the single source of truth for the dataset split metadata.
    Every future module (Downloader, Extractor, Converter, Dataset Loader,
    Trainer, Evaluator, Inference Pipeline) must use this class instead of
    directly reading parquet files.
    """

    def __init__(self, metadata_dir: Path) -> None:
        """Initializes the MetadataManager and loads all dataset splits.

        Args:
            metadata_dir: Path to the directory containing the metadata parquet files.

        Raises:
            TypeError: If metadata_dir is not a pathlib.Path instance.
            FileNotFoundError: If any of the split parquet files are missing.
        """
        if not isinstance(metadata_dir, Path):
            logger.error("metadata_dir must be a pathlib.Path instance, got: %s", type(metadata_dir))
            raise TypeError("metadata_dir must be a pathlib.Path instance")

        self.metadata_dir = metadata_dir

        # Automatically load train, validation, and test splits into private attributes
        self._train_df = self._load_split("train-00000-of-00001.parquet")
        self._val_df = self._load_split("val-00000-of-00001.parquet")
        self._test_df = self._load_split("test-00000-of-00001.parquet")

    def _load_split(self, filename: str) -> pd.DataFrame:
        """Builds file path, validates it exists, reads parquet, and logs success.

        Args:
            filename: The name of the parquet file to load.

        Returns:
            A pandas DataFrame containing the split metadata.

        Raises:
            FileNotFoundError: If the parquet file does not exist at the built path.
        """
        file_path = self.metadata_dir / filename
        if not file_path.exists() or not file_path.is_file():
            logger.error("Metadata file not found: %s", file_path)
            raise FileNotFoundError(f"Metadata file not found: {file_path}")

        try:
            df = pd.read_parquet(file_path)
            logger.info("Successfully loaded metadata split from %s", file_path)
            return df
        except Exception as e:
            logger.error("Failed to read parquet file %s: %s", file_path, e)
            raise

    def _prepare_dataframe(self, df: pd.DataFrame, include50_only: bool) -> pd.DataFrame:
        """Processes a metadata dataframe by optionally filtering, cleaning labels, and resetting index.

        Args:
            df: The raw metadata DataFrame.
            include50_only: If True, filters the dataframe to rows where include_50 is True.

        Returns:
            The processed and copied pandas DataFrame.
        """
        df_copy = df.copy()

        # Optionally filter include_50 == True
        if include50_only:
            if "include_50" in df_copy.columns:
                df_copy = df_copy[df_copy["include_50"] == True]
            else:
                logger.warning("include_50 column not found in DataFrame; skipping filter.")

        # Create clean_label column
        if "label" in df_copy.columns:
            df_copy["clean_label"] = df_copy["label"].apply(self.clean_label)
        else:
            logger.warning("label column not found in DataFrame; cannot create clean_label.")
            df_copy["clean_label"] = None

        # Reset index
        df_copy = df_copy.reset_index(drop=True)
        return df_copy

    @staticmethod
    def clean_label(label: str) -> str:
        """Cleans a label string by removing any numeric prefixes (e.g. '48. Hello' -> 'Hello').

        Args:
            label: The raw label string.

        Returns:
            The cleaned label string.
        """
        if not isinstance(label, str):
            return ""

        # Use regex to strip numeric prefix like "48. " or "48."
        cleaned = re.sub(r"^\d+\.\s*", "", label)
        return cleaned.strip()

    def get_train_split(self, include50_only: bool = True) -> pd.DataFrame:
        """Returns the processed training split dataframe.

        Args:
            include50_only: If True, filters the split to include_50 classes only.

        Returns:
            A processed pandas DataFrame.
        """
        return self._prepare_dataframe(self._train_df, include50_only)

    def get_validation_split(self, include50_only: bool = True) -> pd.DataFrame:
        """Returns the processed validation split dataframe.

        Args:
            include50_only: If True, filters the split to include_50 classes only.

        Returns:
            A processed pandas DataFrame.
        """
        return self._prepare_dataframe(self._val_df, include50_only)

    def get_test_split(self, include50_only: bool = True) -> pd.DataFrame:
        """Returns the processed test split dataframe.

        Args:
            include50_only: If True, filters the split to include_50 classes only.

        Returns:
            A processed pandas DataFrame.
        """
        return self._prepare_dataframe(self._test_df, include50_only)

    def get_class_names(self, include50_only: bool = True) -> list[str]:
        """Returns a sorted list of unique clean class names.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A sorted list of clean labels.
        """
        train_df = self.get_train_split(include50_only)
        val_df = self.get_validation_split(include50_only)
        test_df = self.get_test_split(include50_only)

        unique_classes = set()
        for df in (train_df, val_df, test_df):
            if "clean_label" in df.columns:
                unique_classes.update(df["clean_label"].dropna().unique())

        return sorted(list(unique_classes))

    def get_num_classes(self, include50_only: bool = True) -> int:
        """Returns the number of unique classes.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            The count of unique classes.
        """
        return len(self.get_class_names(include50_only))

    def get_parent_categories(self, include50_only: bool = True) -> list[str]:
        """Returns a sorted list of unique parent categories.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A sorted list of unique parent categories.
        """
        train_df = self.get_train_split(include50_only)
        val_df = self.get_validation_split(include50_only)
        test_df = self.get_test_split(include50_only)

        unique_parents = set()
        for df in (train_df, val_df, test_df):
            if "parent_label" in df.columns:
                unique_parents.update(df["parent_label"].dropna().unique())

        return sorted(list(unique_parents))

    def get_video_paths(self, split: str, include50_only: bool = True) -> list[str]:
        """Returns a list of relative video paths for a given split.

        Args:
            split: The split name ("train", "validation", "test").
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A list of relative video paths.

        Raises:
            ValueError: If the split name is invalid.
        """
        if split == "train":
            df = self.get_train_split(include50_only)
        elif split == "validation":
            df = self.get_validation_split(include50_only)
        elif split == "test":
            df = self.get_test_split(include50_only)
        else:
            logger.error("Invalid split name requested: %s", split)
            raise ValueError(
                f"Invalid split name '{split}'. Supported values are 'train', 'validation', 'test'."
            )

        if "video_path" not in df.columns:
            logger.warning("video_path column not found in split %s", split)
            return []

        return list(df["video_path"].dropna().tolist())

    def get_label_mapping(self, include50_only: bool = True) -> dict[int, str]:
        """Returns a mapping from integer index to clean class name.

        The clean class names are alphabetically sorted first.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A dictionary mapping integers to clean label strings.
        """
        classes = self.get_class_names(include50_only)
        return {i: name for i, name in enumerate(classes)}

    def get_inverse_label_mapping(self, include50_only: bool = True) -> dict[str, int]:
        """Returns a mapping from clean class name to integer index.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A dictionary mapping clean label strings to integers.
        """
        mapping = self.get_label_mapping(include50_only)
        return {name: i for i, name in mapping.items()}

    def get_dataset_statistics(self, include50_only: bool = True) -> dict:
        """Returns a dictionary of statistics for the dataset splits.

        Args:
            include50_only: If True, filters to include_50 classes only.

        Returns:
            A dictionary containing dataset statistics.
        """
        train_df = self.get_train_split(include50_only)
        val_df = self.get_validation_split(include50_only)
        test_df = self.get_test_split(include50_only)

        # Combine all splits for class-wise and category-wise statistics
        combined = pd.concat([train_df, val_df, test_df], ignore_index=True)

        # Compute samples per clean class label
        if "clean_label" in combined.columns:
            class_counts = combined["clean_label"].value_counts().to_dict()
            # Sort the dictionary keys alphabetically for consistent ordering
            samples_per_class = {k: class_counts[k] for k in sorted(class_counts.keys())}
        else:
            samples_per_class = {}

        # Compute parent category distribution
        if "parent_label" in combined.columns:
            parent_counts = combined["parent_label"].dropna().value_counts().to_dict()
            parent_category_distribution = {k: parent_counts[k] for k in sorted(parent_counts.keys())}
        else:
            parent_category_distribution = {}

        return {
            "train_samples": len(train_df),
            "validation_samples": len(val_df),
            "test_samples": len(test_df),
            "num_classes": self.get_num_classes(include50_only),
            "samples_per_class": samples_per_class,
            "parent_category_distribution": parent_category_distribution,
        }
