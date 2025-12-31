"""Model name to ID mapping utilities."""

from loguru import logger


class ModelMapper:
    """Handles mapping between model names and integer IDs."""

    @staticmethod
    def create_mapping(model_names: list[str]) -> dict[str, int]:
        """Create a mapping from model names to integer IDs.

        Args:
            model_names: List of model names (may contain duplicates)

        Returns:
            Dictionary mapping unique model names to IDs (0, 1, 2, ...)
        """
        unique_models = []
        seen = set()

        for name in model_names:
            if name not in seen:
                unique_models.append(name)
                seen.add(name)

        return {name: idx for idx, name in enumerate(unique_models)}

    @staticmethod
    def map_names_to_ids(
        names: list[str], mapping: dict[str, int]
    ) -> list[int]:
        """Convert model names to IDs using the mapping.

        Args:
            names: List of model names
            mapping: Name -> ID mapping

        Returns:
            List of model IDs

        Raises:
            ValueError: If a name is not in the mapping
        """
        result = []
        for name in names:
            if name not in mapping:
                error_msg = f"Model name '{name}' not found in mapping. Available mappings: {list(mapping.keys())}"
                logger.error(f"Model mapping failed: {error_msg}")
                raise ValueError(error_msg)
            result.append(mapping[name])
        return result

    @staticmethod
    def map_ids_to_names(
        ids: list[int], reverse_mapping: dict[int, str]
    ) -> list[str]:
        """Convert model IDs to names using reverse mapping.

        Args:
            ids: List of model IDs
            reverse_mapping: ID -> Name mapping

        Returns:
            List of model names

        Raises:
            ValueError: If an ID is not in the mapping
        """
        result = []
        for model_id in ids:
            if model_id not in reverse_mapping:
                error_msg = f"Model ID {model_id} not found in reverse mapping. Available IDs: {list(reverse_mapping.keys())}"
                logger.error(f"ID to name mapping failed: {error_msg}")
                raise ValueError(error_msg)
            result.append(reverse_mapping[model_id])
        return result
