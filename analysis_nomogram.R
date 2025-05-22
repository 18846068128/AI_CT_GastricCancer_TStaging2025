##################################################
# Nomogram Analysis Script
# Author: [Your Name]
# Date: [Date]
# Version: 1.0
# 
# Inputs:
#   - data/lasso-290.csv       : LASSO筛选后的影像特征
#   - data/clinic_sel.csv      : 临床特征数据
# Outputs:
#   - output/Combined_Nomogram.pdf      : 静态列线图
#   - output/Interactive_Nomogram.html  : 交互式列线图
#   - output/Final_Prediction_Data.csv   : 清洗后的预测数据
#   - output/Model_Summary.txt          : 模型摘要
##################################################

# --------------------------
# 0. Environment Setup
# --------------------------
# 创建输出目录
if (!dir.exists("output")) dir.create("output", showWarnings = FALSE)

# 安装缺失包
required_packages <- c("data.table", "rms", "glmnet", "plotly", "htmlwidgets")
new_packages <- required_packages[!required_packages %in% installed.packages()[,"Package"]]
if(length(new_packages)) install.packages(new_packages)

# 加载依赖包
library(data.table)
library(rms)
library(glmnet)       # 用于LASSO回归
library(htmlwidgets)  # 用于保存交互图表

# --------------------------
# 1. Data Preparation
# --------------------------
# 读取原始数据
features <- fread("data/lasso-290.csv")    # LASSO筛选后的影像特征
clinical <- fread("data/clinic_sel.csv")   # 临床特征数据

# 合并数据集
merged_data <- merge(features, clinical, by = "image_name", all = FALSE)

# 数据完整性校验
stopifnot(
  "合并后数据丢失样本" = nrow(merged_data) == nrow(features),
  "T_stage存在缺失值" = !any(is.na(merged_data$T_stage))
)

# --------------------------
# 2. Radscore Calculation
# --------------------------
set.seed(123)  # 固定随机种子

# 特征矩阵标准化
feature_cols <- grep("^DL-", colnames(merged_data), value = TRUE)
X <- as.matrix(merged_data[, ..feature_cols])
X_scaled <- scale(X)

# LASSO模型训练
cv_fit <- cv.glmnet(
  X_scaled, 
  as.numeric(merged_data$T_stage),
  family = "gaussian", 
  alpha = 1
)

# 计算Radscore
coefs <- as.vector(coef(cv_fit, s = "lambda.min"))[-1]
valid_features <- which(coefs != 0)
merged_data$Radscore <- X_scaled[, valid_features] %*% coefs[valid_features]

# --------------------------
# 3. Data Preprocessing
# --------------------------
# 分类变量转换
merged_data[, `:=`(
  Sex = factor(Sex, levels = c(0, 1), labels = c("Male", "Female")),
  Tumor_location = fcase(
    Tumor_location_L == 1, "Lower",
    Tumor_location_M == 1, "Middle", 
    Tumor_location_U == 1, "Upper"
  ),
  Differentiation = fcase(
    Differentiation_status_poor == 1, "Poor",
    Differentiation_status_moderate == 1, "Moderate",
    Differentiation_status_well == 1, "Well"
  ),
  Lauren = fcase(
    Lauren_intestinal == 1, "Intestinal",
    Lauren_diffuse == 1, "Diffuse",
    Lauren_mixed == 1, "Mixed"
  )
)]

# 清除冗余列
columns_to_remove <- c(
  "Tumor_location_L", "Tumor_location_M", "Tumor_location_U",
  "Differentiation_status_poor", "Differentiation_status_moderate", 
  "Differentiation_status_well", "Lauren_intestinal",
  "Lauren_diffuse", "Lauren_mixed"
)
merged_data[, (columns_to_remove) := NULL]

# --------------------------
# 4. Model Construction
# --------------------------
# 设置数据分布参数
dd <- datadist(merged_data)
options(datadist = "dd")

# 构建有序回归模型
final_model <- orm(
  T_stage ~ Radscore + Age + Tumor_size + CEA + CA19_9 + 
    PD_L1 + Sex + Tumor_location + Differentiation + Lauren,
  data = merged_data,
  x = TRUE, 
  y = TRUE
)

# --------------------------
# 5. Nomogram Generation
# --------------------------
# 自定义刻度函数
smart_scale <- function(x, n = 5) {
  seq(floor(min(x)), ceiling(max(x)), length.out = n)
}

# 生成nomogram对象
nom <- nomogram(
  final_model,
  fun = function(x) plogis(x - 1.6),  # 校准偏移量
  fun.at = seq(0.1, 0.9, by = 0.2),
  lp = FALSE,
  Radscore = smart_scale(merged_data$Radscore),
  Age = seq(30, 80, by = 10),
  Tumor_size = c(2, 5, 8, 12),
  CEA = c(5, 10, 20, 50),
  CA19_9 = c(37, 100, 200, 500),
  PD_L1 = c(0, 1, 5, 10)
)

# --------------------------
# 6. Visualization Output
# --------------------------
# 静态列线图 (PDF)
pdf("output/Combined_Nomogram.pdf", width = 12, height = 8)
plot(nom, 
     col = c("#1f78b4", "#e31a1c"),  # 蓝红配色
     cex.axis = 0.7,
     xfrac = 0.15,
     lplabel = "Linear Predictor")
dev.off()

# 交互式列线图 (HTML)
if (require(plotly)) {
  p <- plot(nom)
  htmlwidgets::saveWidget(
    ggplotly(p) %>% 
      layout(
        title = "Interactive Prediction Nomogram",
        margin = list(t = 50)
      ),
    "output/Interactive_Nomogram.html"
  )
}

# --------------------------
# 7. Result Export
# --------------------------
# 保存清洗后的数据
output_cols <- c(
  "image_name", "Radscore", "T_stage", "Age", "Tumor_size",
  "CEA", "CA19_9", "PD_L1", "Sex", "Tumor_location",
  "Differentiation", "Lauren"
)
write.csv(
  merged_data[, ..output_cols],
  "output/Final_Prediction_Data.csv",
  row.names = FALSE
)

# 保存模型摘要
sink("output/Model_Summary.txt")
print(summary(final_model))
cat("\n--- Radscore Composition ---\n")
print(data.frame(
  Feature = feature_cols[valid_features],
  Coefficient = round(coefs[valid_features], 4)
))
sink()

# --------------------------
# 8. Model Validation
# --------------------------
# Brier评分计算
pred_prob <- predict(final_model, type = "fitted.ind")
true_class <- merged_data$T_stage

brier_scores <- sapply(1:3, function(k) {
  binary_response <- as.numeric(true_class <= k)
  mean((pred_prob[, k] - binary_response)^2)
})
names(brier_scores) <- paste0("P(T≤", 1:3, ")")

# 输出验证结果
cat("\n[Model Validation]\n")
cat("Stratified Brier Scores:\n")
print(round(brier_scores, 3))
cat("Integrated Brier Score:", round(mean(brier_scores), 3))
