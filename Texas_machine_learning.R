library(tidyverse)
library(randomForest)
library(caret)
library(glmnet)
library(MASS)
library(gbm)
library(e1071)
library(klaR)
library(kernlab)
library(kknn)
library(rpart)
library(xgboost)
library(evtree)

X_norm_conc = read.csv("TX X_norm_Conc_1680_obs.csv", header = TRUE, row.names = 1)

FM_class = X_norm_conc$FM_class %>%
  gsub(pattern = "M", replacement = "H", x = .) %>%
  as.factor()

# FM
set.seed(123)
trn_idx_fm = createDataPartition(y = FM_class, p = 0.6, list = FALSE)

# Design matrices
X_norm = X_norm_conc %>%
  dplyr::select(-c(Kernel_ID, AF_class, FM_class, Spec_ID_all))

# Parameters
my_twoClassSummary = function(data, lev = NULL, model = NULL){
  
  a = twoClassSummary(data = data, lev = lev, model = model)
  my_score = ((0.65 * log(0.5 + a["Sens"]) + 0.35 * log(0.5 + a["Spec"])) - log(0.5)) / (log(1.5)-log(0.5))
  out = c(a, my_score)
  names(out) = c("ROC", "Sens", "Spec", "my_score")
  return(out)
}

cv_5 = trainControl(method = "cv", number = 5,
                    classProbs = TRUE, summaryFunction = my_twoClassSummary)

wt_FM = ifelse(test = FM_class[trn_idx_fm] == "L", yes = 1, no = 20)

# Models

# RF
set.seed(123)
mod_norm_rf_fm = train(x = X_norm[trn_idx_fm, ], 
                       y = FM_class[trn_idx_fm],
                       xtest = X_norm[-trn_idx_fm, ],
                       ytest = FM_class[-trn_idx_fm],
                       method = "rf",
                       importance = TRUE,
                       keep.forest = TRUE, 
                       metric = "my_score",
                       trControl = cv_5,
                       classwt = c(9.9*10^3, 1.1*10^-3),
                       tuneGrid = expand.grid(mtry = 39)
)
saveRDS(object = mod_norm_rf_fm, file = "mod_norm_rf_fm.rds")

# SVM
set.seed(123)
mod_norm_svm1_fm = train(x = X_norm[trn_idx_fm, ], 
                         y = FM_class[trn_idx_fm],
                         method = "svmLinearWeights", 
                         metric = "my_score",
                         trControl = cv_5,
                         tuneGrid = expand.grid(cost = c(0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5), 
                                                weight = seq(from = 0.001, to = 0.01, by = 0.001)))
saveRDS(object = mod_norm_svm1_fm, file = "mod_norm_svm1_fm.rds")

set.seed(123)
mod_norm_svm2_fm = train(x = X_norm[trn_idx_fm, ], 
                         y = FM_class[trn_idx_fm],
                         method = "svmPoly", 
                         metric = "my_score",
                         trControl = cv_5,
                         class.weights = c("H" = 20, "L" = 1),
                         tuneGrid = expand.grid(degree = c(1,2,3),
                                                scale = c(0.001, 0.005, 0.01, 0.05, 0.1),
                                                C = c(0.25, 0.5, 1, 2, 5))
)
saveRDS(object = mod_norm_svm2_fm, file = "mod_norm_svm2_fm.rds")

set.seed(123)
mod_norm_svm3_fm = train(x = X_norm[trn_idx_fm, ], 
                         y = FM_class[trn_idx_fm],
                         method = "svmRadialWeights", 
                         metric = "my_score",
                         trControl = cv_5,
                         tuneLength = 27
)
saveRDS(object = mod_norm_svm3_fm, file = "mod_norm_svm3_fm.rds")

# lda
set.seed(123)
mod_norm_lda_fm = train(x = X_norm[trn_idx_fm, ],
                        y = FM_class[trn_idx_fm], 
                        method = "lda",
                        metric = "my_score",
                        prior = c(0.01, 0.99),
                        trControl = cv_5)
saveRDS(object = mod_norm_lda_fm, file = "mod_norm_lda_fm.rds")

# glmnet
set.seed(123)
mod_norm_glmnet_fm = train(x = X_norm[trn_idx_fm, ],
                           y = FM_class[trn_idx_fm], 
                           family = "binomial",
                           weights = wt_FM,
                           method = "glmnet",
                           metric = "my_score",
                           trControl = cv_5, 
                           tuneLength = 25)
saveRDS(object = mod_norm_glmnet_fm, file = "mod_norm_glmnet_fm.rds")

# KNN
set.seed(123)
mod_norm_knn_fm = train(x = X_norm[trn_idx_fm, ],
                           y = FM_class[trn_idx_fm], 
                           method = "kknn",
                           metric = "my_score",
                           trControl = cv_5, 
                           tuneLength = 27)
saveRDS(object = mod_norm_knn_fm, file = "mod_norm_knn_fm.rds")

# GBM
set.seed(123)
mod_norm_gbm_fm = train(x = X_norm[trn_idx_fm, ],
                        y = FM_class[trn_idx_fm], 
                        method = "gbm",
                        metric = "my_score",
                        trControl = cv_5, 
                        weights = wt_FM,
                        tuneLength = 32, 
                        verbose = FALSE)
saveRDS(object = mod_norm_gbm_fm, file = "mod_norm_gbm_fm.rds")

data_trn = cbind(FM_class, X_norm)

# xgboost
set.seed(123)
mod_norm_xgboost_fm = train(x = X_norm[trn_idx_fm, ],
                        y = FM_class[trn_idx_fm], 
                        method = "xgbTree",
                        metric = "my_score",
                        trControl = cv_5,
                        weights = wt_FM,
                        tuneLength = 10)
saveRDS(object = mod_norm_xgboost_fm, file = "mod_norm_xgboost_fm.rds")

# Evolution tree
set.seed(123)
mod_norm_evtree_fm = train(x = X_norm[trn_idx_fm, ],
                            y = FM_class[trn_idx_fm], 
                            method = "evtree",
                            metric = "my_score",
                            trControl = cv_5,
                            tuneGrid = expand.grid(alpha = seq(from = 0.001, to = 0.1, by = 0.006)))
saveRDS(object = mod_norm_evtree_fm, file = "mod_norm_evtree_fm.rds")
