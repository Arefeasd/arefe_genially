"""Create summary table for patient comorbidities and medical history

This module creates a table with a row per patient
and one column per category of comorbidity
(furnished by Nancy Hospital team) located in the companion
cim10_nancy.json file

Remark:
    - we focus on patients with a unique inclusion stay
    - and stays at the hospital within the two year before the inclusion stay
    - medical history categories were furnished by Nancy Hospital team
    - a set of corresponding codes (cim10) were suggested that MUST be refined and validated by clinicians
    - basic demographics were also included (age at inclusion time, gender)
    - clinical scores were added (Charlson Quan, Hospital Frailty Index)
    - clinical scores were computed using pycomorb package

TODO:
    + add other terminologies (e.g. ccam or lls ?)

"""
import pathlib
import json
import re

import polars as pl
from pycomorb import comorbidity

"""
This function checks a provided list of columns and converts any column 
containing string-based date/time data into the standard Polars datetime format.
It uses 'strict=False' to ensure that invalid or malformed dates do not 
crash the data loading pipeline.
"""
def cast_datetime_cols(df, datetime_cols):

    cols_to_cast = [c for c in datetime_cols if df[c].dtype == pl.String]
    if cols_to_cast:
        df = df.with_columns(
            pl.col(c).str.to_datetime(format=None, strict=False) for c in cols_to_cast
        )
    return df

"""
Automatically identifies all columns ending with '_id' using regex 
and casts them to Utf8 (string) format. This ensures consistency 
when dealing with identifiers that might contain leading zeros or 
alphanumeric characters, preventing potential issues during data merging.
"""
def cast_id_cols(df):

    df = df.with_columns(pl.col("^.*_id$").cast(pl.Utf8))
    return df

"""
Loads patient demographic data from a Parquet file, selects essential 
columns, and cleans the data by standardizing IDs and date formats. 
It also merges internal and external death dates into a single 'death_date' 
column using coalesce, ensuring accurate mortality tracking for the cohort.
"""
def load_patients(path_patients):
    patients = (
        pl.scan_parquet(path_patients)
        .select(
            [
                "patient_id",
                "gender",
                "birth_date",
                "internal_death_date",
                "external_death_date",
            ]
        )
        .collect()
    )
    # set the correct types for columns
    date_cols = ["birth_date", "internal_death_date", "external_death_date"]
    patients = cast_id_cols(patients)
    patients = cast_datetime_cols(patients, date_cols)
    # consolidate death_age
    # Consolidate death dates by prioritizing external records, then internal records
    patients = patients.with_columns(
        pl.coalesce(["external_death_date", "internal_death_date"]).alias("death_date")
    ).select(["patient_id", "gender", "birth_date", "death_date"])
    return patients

"""
Loads hospital stay records, ensuring uniqueness for each patient-stay combination.
It filters for essential columns including the inclusion flag, and standardizes 
ID and datetime formats to ensure consistency with demographic data. This 
table is crucial for defining patient trajectories and time-based medical history.
"""
def load_hosp(path_stays):

    hosp = (
        pl.scan_parquet(path_stays)
        .select(
            [
                "patient_id",
                "hosp_id",
                "hosp_admission_datetime",
                "hosp_discharge_datetime",
                "inclusion_flag",
            ]
        )
        # remove duplicates based on patient, hospital, and inclusion status,
        .unique(subset=["patient_id", "hosp_id", "inclusion_flag"])
        .collect()
    )
    hosp = cast_id_cols(hosp)
    date_cols = ["hosp_admission_datetime", "hosp_discharge_datetime"]
    hosp = cast_datetime_cols(hosp, date_cols)
    return hosp

def load_cim10(path_structured_data, selected_cim10_codes):
    """
    Load only the CIM10 diagnosis rows that are useful for the comorbidity table.

    Here, we keep:
    - CIM10 codes explicitly listed in cim10_nancy.json
    - cancer diagnosis codes from C00 to C97, because "Cancer actif" is defined as C00-C97
    """

    cim10 = (
        pl.scan_parquet(path_structured_data)
        .filter(pl.col("terminology_code") == "cim10")
        .select(["patient_id", "hosp_id", "concept_code"])
        .filter(
            pl.col("concept_code").is_in(selected_cim10_codes)
            |
            pl.col("concept_code").str.contains(r"^C([0-8][0-9]|9[0-7])")
        )
        .unique()
        .collect()
    )

    cim10 = cast_id_cols(cim10)
    return cim10

"""
Filters the hospital stays to retain only patients who have exactly one 
'inclusion_flag' event throughout the study period. This ensures that 
longitudinal analyses (like survival or care pathway modeling) are not 
skewed by patients with multiple recurring qualifying admissions.
"""
def get_unique_inclusion_hosp(hosp):

    unique_inclusion_hosp = (
        hosp.filter(pl.col("inclusion_flag") == 1)
        .group_by("patient_id")
        .agg(pl.len().alias("nb_hosp_inclusion"))
        .filter(pl.col("nb_hosp_inclusion") == 1)
        .select(["patient_id"])
        .join(hosp, on="patient_id", how="inner")
        .select(hosp.columns)
    )
    return unique_inclusion_hosp

"""
Extracts all historical hospital stays for patients that are NOT 
the inclusion stay (i.e., inclusion_flag == 0). This allows for the 
analysis of a patient's medical history and care trajectory in the 
two years prior to their qualifying admission.
"""
def get_preceding_hosp(hosp):
    return hosp.filter(pl.col("inclusion_flag") == 0)

"""
Maps clinical diagnosis codes (ICD-10) to clinically meaningful comorbidity 
categories. It performs a horizontal sum across related codes for each category; 
if any code is present (sum > 0), it assigns a 1 (binary flag), otherwise 0.
Includes specialized regex handling for broad categories like 'Active Cancer'.
"""
def get_comorbidities(df: pl.DataFrame, cim10: dict) -> pl.DataFrame:
    """ """
    all_columns = set(df.columns)
    expressions = []

    for categorie, codes_dict in cim10.items():
        # Special case : Active Cancer → range cim10 codes C00–C97
        if categorie == "Cancer actif":
            codes_presents = [
                col
                for col in all_columns
                if re.match(r"^C\d{2}", col) and 0 <= int(col[1:3]) <= 97
            ]
        else:
            codes_presents = [code for code in codes_dict.keys() if code in all_columns]

        if not codes_presents:
            # aucun code de la catégorie présent dans le DataFrame
            expressions.append(pl.lit(0).cast(pl.Int8).alias(categorie))
            continue

        # 1 si la somme des codes de la catégorie > 0
        expr = (
            (pl.sum_horizontal([pl.col(c) for c in codes_presents]) > 0)
            .cast(pl.Int8)
            .alias(categorie)
        )

        expressions.append(expr)

    return df.with_columns(expressions).select(["patient_id"] + expressions)



if __name__ == "__main__":
    
    # Raw data folder on the ODH server
    DATA_PATH = pathlib.Path("/data/data")

    # Folder containing this script and the cim10_nancy.json mapping file
    COMORBIDITY = pathlib.Path("/data/workspace/arefe_genially/scripts/comorbidity")

    # Output folder in the project workspace
    OUTPUT_PATH = pathlib.Path("/data/workspace/arefe_genially/data")
    # tabular files 
    path_stays = DATA_PATH/"stay.parquet"
    path_patients = DATA_PATH/ "patient.parquet"
    path_structured_data = DATA_PATH/ "document_data.parquet" 
    path_comorbidities_file = COMORBIDITY/ "cim10_nancy.json"
    path_output_file =  OUTPUT_PATH / "final_patient_data.parquet"


    
    # we restrict our analysis with patients with only one inclusion hospitalisation
    # it is a bias as we expect these patients to be less severe
    # but ease the computation of comorbidities and medical history retrieval
    hosp = load_hosp(path_stays)
    unique_inclusion_hosp = get_unique_inclusion_hosp(hosp)
    preceding_hosp = get_preceding_hosp(hosp)
    # get hosp containing medical history
    # From all historical hospital stays (preceding_hosp), keep only those 
    # where the patient_id belongs to our list of eligible, unique-inclusion patients.
    interest_hosp = preceding_hosp.join(
        unique_inclusion_hosp.select("patient_id"), on="patient_id", how="inner"
    )

    # Load the clinical comorbidity mapping before loading CIM10 data.
    # This allows us to restrict document_data to only the CIM10 codes needed later.
    with open(path_comorbidities_file, encoding="utf-8") as f:
        cim10_nancy = json.load(f)

    # Extract the explicit CIM10 codes listed in cim10_nancy.json.
    # The empty key used for "Cancer actif" is excluded here because cancer is handled separately as C00-C97.

    selected_cim10_codes = sorted({
        code
        for codes_dict in cim10_nancy.values()
        for code in codes_dict.keys()
        if code != ""
    })

    # Load only useful CIM10 rows instead of loading all CIM10 rows from document_data.
    cim10 = load_cim10(path_structured_data, selected_cim10_codes)

    # Build a patient-level CIM10 indicator table.
    # Rows correspond to patients and columns correspond only to selected CIM10 codes
    # used for the Nancy comorbidity mapping.
    interest_hosp_cim10 = cim10.join(
        interest_hosp, on=["patient_id", "hosp_id"], how="inner"
    )
    # Reshape the data from long format (patient-concept pairs) to wide format (feature matrix),
    # where each row represents a unique patient and each column serves as a binary indicator 
    # for a specific medical concept (1 if the code is present, 0 otherwise).
    patient_cim10 = (
        interest_hosp_cim10.select(["patient_id", "concept_code"])
        .with_columns(pl.lit(1).alias("value"))
        .pivot(
            on="concept_code",
            values="value",
            index="patient_id",  # rows : one per patient_id
            aggregate_function="first",  # Indicator could be changed by `count` if needed
        )
        .with_columns(
        pl.col("patient_id").cast(pl.Utf8),
        pl.exclude("patient_id").fill_null(0).cast(pl.Int64)
    )
    .sort("patient_id")
    )

    # infer comorbidities by categories furnished by nancy team
    patient_comorbidities = get_comorbidities(patient_cim10, cim10_nancy)

    # Add Demographics (Sex, Age at inclusion)
    patient = load_patients(path_patients)

    patient_demo_full = patient.join(
        unique_inclusion_hosp, on="patient_id", how="inner"
    )
    patient_demo_full = patient_demo_full.with_columns(
        (
            (pl.col("hosp_admission_datetime") - pl.col("birth_date")).dt.total_days()
            / 365.25
        )
        .floor()
        .cast(pl.Int32)
        .alias("age_at_inclusion")
    )

    # data_before_inclusion = patient_demo_full.join(
    #     interest_hosp_cim10, on="patient_id", how="inner"
    # )

    # print(data_before_inclusion.shape)

    # print(
    #     data_before_inclusion
    #     .select(pl.col("concept_code").n_unique())
    # )


    # charlson = comorbidity(
    #     score="charlson",
    #     df=data_before_inclusion,
    #     id_col="patient_id",
    #     code_col="concept_code",
    #     age_col="age_at_inclusion",
    #     implementation="quan",
    # )
    # hfrs = comorbidity(
    #     score="hfrs",
    #     df=data_before_inclusion,
    #     id_col="patient_id",
    #     code_col="concept_code",
    #     age_col="age_at_inclusion",
    # )

    # # Convert pycomorb outputs to Polars DataFrames if needed.
    # if not isinstance(charlson, pl.DataFrame):
    #     charlson = pl.from_pandas(charlson)

    # if not isinstance(hfrs, pl.DataFrame):
    #     hfrs = pl.from_pandas(hfrs)

    # # Ensure patient_id has the same type before joining.
    # charlson = cast_id_cols(charlson)
    # hfrs = cast_id_cols(hfrs)

    # patient_final = (patient_demo_full.join(
    #     patient_comorbidities, on="patient_id", how="inner"
    # ).join(charlson, on=['patient_id'], how='inner')
    # .join(hfrs,on=['patient_id'], how='inner')) 

    patient_final = (patient_demo_full.join(
         patient_comorbidities, on="patient_id", how="inner"))

    patient_final.write_parquet(
    path_output_file,
    compression="snappy",  # ou "gzip", "zstd", None
)
