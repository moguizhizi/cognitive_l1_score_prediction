from enum import Enum


class ColumnName(str, Enum):
    HOSPITAL_ID = "医院id"
    HOSPITAL_NAME = "医院"
    PATIENT_NAME = "患者姓名"
    PATIENT_ID = "患者编码"
    GENDER = "性别"
    BIRTH_DATE = "出生日期"
    AGE_AT_TRAINING_START = "开始训练的年龄"
    AGE_GROUP = "年龄组"
    DISEASE_LABEL = "疾病标签"
    DEPARTMENT = "科室"
    FIRST_ASSESSMENT_MEAN_SCORE = "首次测评脑能力均值(任务40-160均值)"
    PRETEST_GROUP = "前测组别"
    FIRST_TRAINING_DATE = "首个应训练日日期"
    TRAINING_WEEK = "训练周数"
    QUALIFIED_TRAINING_DAYS = "该周达标天数"
    PERCEPTION = "感知觉"
    ATTENTION = "注意力"
    MEMORY = "记忆力"
    EXECUTIVE_FUNCTION = "执行功能"
