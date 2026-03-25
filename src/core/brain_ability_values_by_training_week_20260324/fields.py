from src.core.brain_ability_values_by_training_week_20260324.constants import (
    ColumnName,
)


REQUIRED_COLUMNS = [
    ColumnName.PATIENT_NAME,
    ColumnName.PATIENT_ID,
    ColumnName.TRAINING_WEEK,
    ColumnName.PERCEPTION,
    ColumnName.ATTENTION,
    ColumnName.MEMORY,
    ColumnName.EXECUTIVE_FUNCTION,
]

NUMERIC_COLUMNS = [
    ColumnName.FIRST_ASSESSMENT_MEAN_SCORE,
    ColumnName.TRAINING_WEEK,
    ColumnName.QUALIFIED_TRAINING_DAYS,
    ColumnName.PERCEPTION,
    ColumnName.ATTENTION,
    ColumnName.MEMORY,
    ColumnName.EXECUTIVE_FUNCTION,
]

DATE_COLUMNS = [
    ColumnName.BIRTH_DATE,
    ColumnName.FIRST_TRAINING_DATE,
]
