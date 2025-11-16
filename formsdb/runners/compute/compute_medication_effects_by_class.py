#!/usr/bin/env python
"""
Compile medication effects - active medications - during various modality timepoints.

Uses medication data compiled earlier in the pipeline, to infer medication effects.
"""
import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent
ROOT = None
for parent in file.parents:
    if parent.name == "ampscz-formsdb":
        ROOT = parent
sys.path.append(str(ROOT))

# remove current directory from path
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import logging
import multiprocessing

# import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from psycopg2.extensions import AsIs, register_adapter
from rich.logging import RichHandler

from formsdb import data
from formsdb.helpers import db, utils

# Addresses:
# sqlalchemy.exc.ProgrammingError: (psycopg2.ProgrammingError) can't adapt type 'numpy.int64'
register_adapter(np.int64, AsIs)

MODULE_NAME = "formsdb.runners.compute.medication_effects_by_class"

console = utils.get_console()

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.DEBUG,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)


def get_all_med_classes(config_file: Path) -> List[str]:
    """
    Fetches all unique medication classes from the database.

    Args:
        config_file (Path): Path to the configuration file.

    Returns:
        List[str]: List of unique medication classes.
    """
    query = """
        SELECT DISTINCT med_class FROM forms_derived.medication_data
    """
    df = db.execute_sql(
        config_file=config_file,
        query=query,
    )
    all_med_classes = df["med_class"].tolist()
    all_med_classes.sort()
    return all_med_classes


def get_subject_medication_effects(subject_id: str, config_file: Path) -> pd.DataFrame:
    """
    Fetches medication effects for a given subject from the database.

    Args:
        subject_id (str): The ID of the subject.
        config_file (Path): Path to the configuration file.

    Returns:
        pd.DataFrame: DataFrame containing medication effects.
    """
    query = f"""
        SELECT * FROM forms_derived.medication_effect
        WHERE subject_id = '{subject_id}'
    """
    df = db.execute_sql(
        config_file=config_file,
        query=query,
    )
    return df


def get_subject_medication_effects_info_by_class(
    config_file: Path,
    subject_id: str,
) -> List[Dict[str, Any]]:
    """
    Get medication effect information for a subject, grouped by medication class.

    Args:
        config_file (Path): Path to the configuration file.
        subject_id (str): The ID of the subject.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing medication effect
            information for the subject, grouped by medication class.
    """
    all_med_classes = get_all_med_classes(config_file=config_file)
    subject_medication_df = get_subject_medication_effects(
        subject_id=subject_id, config_file=config_file
    )
    modality_date_df = subject_medication_df[
        ["modality", "timepoint", "date", "days_from_consent"]
    ].drop_duplicates()

    medication_effects_by_class: List[Dict[str, Any]] = []

    for _, row in modality_date_df.iterrows():
        modality = row["modality"]
        timepoint = row["timepoint"]
        date = row["date"]
        days_from_consent = row["days_from_consent"]

        # Filter the DataFrame for the current modality, timepoint
        filtered_df: pd.DataFrame = subject_medication_df[
            (subject_medication_df["modality"] == modality)
            & (subject_medication_df["timepoint"] == timepoint)
        ]

        for med_class in all_med_classes:
            med_class_df: pd.DataFrame = filtered_df[
                filtered_df["med_class"] == med_class
            ]
            if med_class_df.empty:
                result = {
                    "subject_id": subject_id,
                    "modality": modality,
                    "timepoint": timepoint,
                    "date": date,
                    "days_from_consent": days_from_consent,
                    "med_class": med_class,
                    "days_since_last_taken": None,
                    "current_use": 0,
                    "ever_used": 0,
                }
            else:
                if (med_class_df["current_use"] == 1).any():
                    current_use = 1
                elif med_class_df["current_use"].isna().any():
                    current_use = None
                else:
                    current_use = 0

                if (med_class_df["ever_used"] == 1).any():
                    ever_used = 1
                elif med_class_df["ever_used"].isna().any():
                    ever_used = None
                else:
                    ever_used = 0

                def sum_or_none(series: pd.Series) -> Optional[Union[int, float]]:
                    """
                    Sums the series if it does not contain any null values,
                    otherwise returns None.
                    """
                    return None if series.isnull().any() else series.sum()

                if med_class == "ANTIPSYCHOTIC":
                    ap_equivalent_drug_dose_prescribed = sum_or_none(
                        med_class_df["ap_equivalent_drug_dose_prescribed"]
                    )
                    ap_equivalent_drug_dose_estimated_taken = sum_or_none(
                        med_class_df["ap_equivalent_drug_dose_estimated_taken"]
                    )
                    ap_equivalent_drug_dose_prescribed_leucht = sum_or_none(
                        med_class_df["ap_equivalent_drug_dose_prescribed_leucht"]
                    )
                    ap_equivalent_drug_dose_estimated_taken_leucht = sum_or_none(
                        med_class_df["ap_equivalent_drug_dose_estimated_taken_leucht"]
                    )
                else:
                    ap_equivalent_drug_dose_prescribed = None
                    ap_equivalent_drug_dose_estimated_taken = None
                    ap_equivalent_drug_dose_prescribed_leucht = None
                    ap_equivalent_drug_dose_estimated_taken_leucht = None

                if med_class == "BENZODIAZEPINE":
                    bd_equivalent_drug_dose_prescribed = sum_or_none(
                        med_class_df["bd_equivalent_drug_dose_prescribed"]
                    )
                    bd_equivalent_drug_dose_estimated_taken = sum_or_none(
                        med_class_df["bd_equivalent_drug_dose_estimated_taken"]
                    )
                else:
                    bd_equivalent_drug_dose_prescribed = None
                    bd_equivalent_drug_dose_estimated_taken = None

                prescribed_equivalent_drug_dose_for_day = sum_or_none(
                    med_class_df["prescribed_equivalent_drug_dose_for_day"]
                )
                complied_equivalent_drug_dose_for_day = sum_or_none(
                    med_class_df["complied_equivalent_drug_dose_for_day"]
                )

                days_since_last_taken_values = med_class_df[
                    "days_since_last_taken"
                ].dropna()
                if not days_since_last_taken_values.empty:
                    days_since_last_taken = days_since_last_taken_values.min()
                else:
                    days_since_last_taken = None

                result = {
                    "subject_id": subject_id,
                    "modality": modality,
                    "timepoint": timepoint,
                    "date": date,
                    "days_from_consent": days_from_consent,
                    "med_class": med_class,
                    "days_since_last_taken": days_since_last_taken,
                    "current_use": current_use,
                    "ever_used": ever_used,
                    "ap_equivalent_drug_dose_prescribed": ap_equivalent_drug_dose_prescribed,
                    "ap_equivalent_drug_dose_estimated_taken": ap_equivalent_drug_dose_estimated_taken,
                    "ap_equivalent_drug_dose_prescribed_leucht": ap_equivalent_drug_dose_prescribed_leucht,
                    "ap_equivalent_drug_dose_estimated_taken_leucht": ap_equivalent_drug_dose_estimated_taken_leucht,
                    "bd_equivalent_drug_dose_prescribed": bd_equivalent_drug_dose_prescribed,
                    "bd_equivalent_drug_dose_estimated_taken": bd_equivalent_drug_dose_estimated_taken,
                    "prescribed_equivalent_drug_dose_for_day":
                        prescribed_equivalent_drug_dose_for_day,
                    "complied_equivalent_drug_dose_for_day":
                        complied_equivalent_drug_dose_for_day,
                }

            medication_effects_by_class.append(result)

    return medication_effects_by_class


def process_subject_wrapper(params: Tuple[Path, str]) -> List[Dict[str, Any]]:
    """
    Process a single subject's medication effect information.

    Args:
        params (Tuple[Path, str]): A tuple containing the config file path and subject ID.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing medication effect
            information for the subject.
    """
    config_file, subject_id = params
    return get_subject_medication_effects_info_by_class(
        config_file=config_file, subject_id=subject_id
    )


def compile_medication_effects(
    config_file: Path,
) -> None:
    """
    Compile medication effects for all subjects.
    """
    medication_effect_raw_data: List[Dict[str, Any]] = []

    subjects = data.get_all_subjects(config_file=config_file)

    # max_count = 25
    # subjects = random.sample(subjects, min(max_count, len(subjects)))

    num_processes = multiprocessing.cpu_count() // 4
    logger.info(f"Using {num_processes} processes for parallel computation.")
    params = [(config_file, subject_id) for subject_id in subjects]

    with multiprocessing.Pool(processes=int(num_processes)) as pool:
        with utils.get_progress_bar() as progress:
            task = progress.add_task("Processing subjects...", total=len(params))
            for returned_result in pool.imap_unordered(process_subject_wrapper, params):
                medication_effect_raw_data.extend(returned_result)
                progress.update(task, advance=1)

    medication_effect_raw_data_df = pd.DataFrame(medication_effect_raw_data)

    int_cols = [
        "days_from_consent",
        "ap_equivalent_drug_dose_prescribed",
        "ap_equivalent_drug_dose_estimated_taken",
        "ap_equivalent_drug_dose_prescribed_leucht",
        "ap_equivalent_drug_dose_estimated_taken_leucht",
        "bd_equivalent_drug_dose_prescribed",
        "bd_equivalent_drug_dose_estimated_taken",
        "current_use",
        "ever_used",
    ]

    for col in int_cols:
        medication_effect_raw_data_df[col] = (
            pd.to_numeric(medication_effect_raw_data_df[col], errors="coerce")
            .round(0)
            .astype("Int64")
        )

    # Cast date columns to datetime
    date_cols = ["date"]

    for col in date_cols:
        medication_effect_raw_data_df[col] = pd.to_datetime(
            medication_effect_raw_data_df[col], errors="coerce"
        )

    medication_effect_raw_data_df = medication_effect_raw_data_df.convert_dtypes()

    logger.info(
        f"Number of medication effect records: {len(medication_effect_raw_data_df)}"
    )
    logger.info(
        "Overwriting the medication_effect table in the database with the new data."
    )
    db.df_to_table(
        config_file=config_file,
        df=medication_effect_raw_data_df,
        table_name="medication_effect_by_class",
        schema="forms_derived",
        if_exists="replace",
    )

    return


if __name__ == "__main__":
    console.rule(f"[bold red]{MODULE_NAME}")

    config_file = utils.get_config_file_path()
    console.print(f"Using config file: {config_file}")

    utils.configure_logging(
        config_file=config_file, module_name=MODULE_NAME, logger=logger
    )

    logger.info("Compiling medication effects...")
    compile_medication_effects(config_file=config_file)
    logger.info("Done!")
