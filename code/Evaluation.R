require(tidyverse)
require(readxl)
require(caret)
library(pROC)

rm(list=ls())

excel_path = "/home/panky/PROJ/LLM/Results/patient_results.xlsx"
df = read_excel(excel_path)

data = df %>%
  mutate(response_2 = if_else(
    response=="pCR", 1, 0
  )) %>%
  mutate(predict_2 = if_else(
    predict=="pCR", 1, 0
  ))

###########################
CURVE = roc(data$response_2, data$predict_2)

conf_matrix = caret::confusionMatrix(
  data = as.factor(data$predict_2),
  reference = as.factor(data$response_2),
  positive = "1"
)

conf_matrix$byClass
conf_matrix$overall

AUC = sprintf("%0.3f", CURVE$auc)
ACC = sprintf("%0.3f", conf_matrix$byClass[11])
SEN = sprintf("%0.3f", conf_matrix$byClass[1])
SPE = sprintf("%0.3f", conf_matrix$byClass[2])
PPV = sprintf("%0.3f", conf_matrix$byClass[3])
NPV = sprintf("%0.3f", conf_matrix$byClass[4])
F1 = sprintf("%0.3f", conf_matrix$byClass[7])

table_2 = matrix("", 6, 8) %>%
  as.data.frame() %>%
  set_names("Model", "AUROC", "ACC", "SEN", "SPE", "PPV", "NPV", "F1")
table_2[1, 1] = "Qwen3-4B-Thinking-2507"
table_2[1, 2:8] = c(AUC, ACC, SEN, SPE, PPV, NPV, F1)


