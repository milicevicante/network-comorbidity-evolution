"""
Data inventory module for discovering and cataloging available GEXF graph files.

This module scans the data directory for GEXF files and extracts stratum identifiers
(sex, variant, age group, period) from standardized filenames.

Filename patterns:
- Age-stratified: Graph_{Sex}_{Variant}_Age_{N}.gexf
- Year-stratified: Graph_{Sex}_{Variant}_Year_{YYYY}.gexf
"""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union


@dataclass
class GraphStratum:
    """
    Represents a single graph file with its stratum identifiers.

    Attributes:
        path: Full path to the GEXF file
        sex: 'male' or 'female'
        variant: 'blocks', 'chronic', or 'icd'
        age_group: Integer 1-8 if age-stratified, None otherwise
        period: Year string (e.g., '2003') if year-stratified, None otherwise
        filename: Original filename without path
    """
    path: Path
    sex: str
    variant: str
    age_group: Optional[int]
    period: Optional[str]
    filename: str

    def __repr__(self) -> str:
        strat = f"age={self.age_group}" if self.age_group else f"period={self.period}"
        return f"GraphStratum({self.sex}/{self.variant}/{strat})"


# Regex patterns for parsing GEXF filenames
# Pattern: Graph_{Sex}_{Variant}_Age_{N}.gexf or Graph_{Sex}_{Variant}_Year_{YYYY}.gexf
AGE_PATTERN = re.compile(
    r"Graph_(?P<sex>Male|Female)_(?P<variant>Blocks|Chronic|ICD)_Age_(?P<age>\d+)\.gexf$",
    re.IGNORECASE
)
YEAR_PATTERN = re.compile(
    r"Graph_(?P<sex>Male|Female)_(?P<variant>Blocks|Chronic|ICD)_Year_(?P<year>\d{4})\.gexf$",
    re.IGNORECASE
)


def parse_gexf_filename(filename: str) -> Optional[dict]:
    """
    Parse a GEXF filename to extract stratum identifiers.

    Args:
        filename: Name of the GEXF file (not full path)

    Returns:
        Dictionary with keys {sex, variant, age_group, period} or None if no match.
        Only one of age_group or period will be set; the other will be None.
    """
    # Try age-stratified pattern first
    match = AGE_PATTERN.match(filename)
    if match:
        return {
            "sex": match.group("sex").lower(),
            "variant": match.group("variant").lower(),
            "age_group": int(match.group("age")),
            "period": None,
        }

    # Try year-stratified pattern
    match = YEAR_PATTERN.match(filename)
    if match:
        return {
            "sex": match.group("sex").lower(),
            "variant": match.group("variant").lower(),
            "age_group": None,
            "period": match.group("year"),
        }

    return None


def inventory_gexf_files(
    data_root: Union[str, Path],
    gexf_subdir: str = "4.Graphs-gexffiles",
) -> List[GraphStratum]:
    """
    Scan the data directory and inventory all available GEXF files.

    Args:
        data_root: Root directory containing data folders (e.g., ../Data)
        gexf_subdir: Subdirectory name containing GEXF files

    Returns:
        List of GraphStratum objects, one per discovered GEXF file.
        Files that don't match expected naming patterns are skipped.
    """
    gexf_dir = Path(data_root) / gexf_subdir

    if not gexf_dir.exists():
        raise FileNotFoundError(f"GEXF directory not found: {gexf_dir}")

    strata = []
    for filepath in sorted(gexf_dir.glob("*.gexf")):
        parsed = parse_gexf_filename(filepath.name)
        if parsed:
            strata.append(GraphStratum(
                path=filepath,
                sex=parsed["sex"],
                variant=parsed["variant"],
                age_group=parsed["age_group"],
                period=parsed["period"],
                filename=filepath.name,
            ))

    return strata


def find_graph_file(
    data_root: Union[str, Path],
    sex: str,
    variant: str,
    age_group: Optional[int] = None,
    period: Optional[str] = None,
    gexf_subdir: str = "4.Graphs-gexffiles",
) -> Optional[Path]:
    """
    Find a specific GEXF file by stratum identifiers.

    Args:
        data_root: Root directory containing data folders
        sex: 'male' or 'female'
        variant: 'blocks', 'chronic', or 'icd'
        age_group: Integer 1-8 for age-stratified files
        period: Year string for year-stratified files
        gexf_subdir: Subdirectory name containing GEXF files

    Returns:
        Path to the matching GEXF file, or None if not found.

    Raises:
        ValueError: If neither age_group nor period is specified
    """
    if age_group is None and period is None:
        raise ValueError("Must specify either age_group or period")

    # Normalize inputs
    sex = sex.lower()
    variant = variant.lower()

    # Construct expected filename
    sex_cap = sex.capitalize()
    variant_cap = variant.upper() if variant == "icd" else variant.capitalize()

    if age_group is not None:
        filename = f"Graph_{sex_cap}_{variant_cap}_Age_{age_group}.gexf"
    else:
        filename = f"Graph_{sex_cap}_{variant_cap}_Year_{period}.gexf"

    filepath = Path(data_root) / gexf_subdir / filename

    return filepath if filepath.exists() else None


def summarize_inventory(strata: List[GraphStratum]) -> dict:
    """
    Generate a summary of the inventoried data.

    Args:
        strata: List of GraphStratum objects from inventory_gexf_files()

    Returns:
        Dictionary with summary statistics:
        - total: Total number of files
        - by_sex: Count by sex
        - by_variant: Count by variant
        - age_stratified: Count of age-stratified files
        - year_stratified: Count of year-stratified files
        - combinations: List of unique (sex, variant) combinations
    """
    summary = {
        "total": len(strata),
        "by_sex": {},
        "by_variant": {},
        "age_stratified": 0,
        "year_stratified": 0,
        "combinations": set(),
    }

    for s in strata:
        # Count by sex
        summary["by_sex"][s.sex] = summary["by_sex"].get(s.sex, 0) + 1

        # Count by variant
        summary["by_variant"][s.variant] = summary["by_variant"].get(s.variant, 0) + 1

        # Count stratification type
        if s.age_group is not None:
            summary["age_stratified"] += 1
        else:
            summary["year_stratified"] += 1

        # Track combinations
        summary["combinations"].add((s.sex, s.variant))

    summary["combinations"] = sorted(summary["combinations"])

    return summary
