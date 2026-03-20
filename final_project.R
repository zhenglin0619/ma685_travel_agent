################################################################################
# Final Project: Vacation Recommendation and Travel Agent Engine/System
# Base Models: KNN, DT, RF, SVM, MLR thresh, KMC, NB, XGBoost
################################################################################
source('src/knn.R')
source('src/kmc.R')
library(rpart)
library(randomForest)
library(e1071)
library(xgboost)
library(lightgbm)

################################################################################
### 1. Data Preprocessing
users = read.csv('src/Vacation_UsersData.csv')
dests = read.csv('src/Vacation_DestinationData.csv')

users$ClimatePreference = factor(users$ClimatePreference)
users$TravelStylePreference = factor(users$TravelStylePreference)
users$TravelCompanionPreference = factor(users$TravelCompanionPreference)
users$FavoriteColor = factor(users$FavoriteColor)
users$LocalSportsTeamRecord = factor(users$LocalSportsTeamRecord)

set.seed(1234)
n = nrow(users)
ptrain = 0.65
ntrain = floor(n*ptrain)
idx = seq(1, n, 1)
itrain = sample(idx, ntrain, replace=F)
itest = setdiff(idx, itrain)
Xtrain = users[itrain,]
Xtest = users[itest,]

preprocess = function(Xtrain, dests) {
  parse_trips = function(trips) {
    parts = unlist(strsplit(trips, ", "))
    
    loc = c()
    rat = c()
    for (p in parts) {
      reg = regexec("Destination_([0-9]+)\\s*-\\s*([0-9]+)", p)
      output = unlist(regmatches(p, reg))
      loc = c(loc, as.integer(output[2]))
      rat = c(rat, as.integer(output[3]))
    }
    return(data.frame(LocationID = loc, Rating = rat))
  }
  
  Xtrain_list = lapply(1:nrow(Xtrain), function(i) {
    trips = parse_trips(Xtrain$PastTripsWithRatings[i])
    base = Xtrain[i,]
    base = base[rep(1, nrow(trips)), , drop=F]
    return(cbind(base, trips))
  })
  
  Xtrain_all = do.call(rbind, Xtrain_list)
  Xtrain_merge = merge(Xtrain_all, dests, by = "LocationID")
  
  drop_cols = c("UserID", "PastTripsWithRatings", "Rating", "CityName", "LocationID")
  feature_cols = setdiff(names(Xtrain_merge), drop_cols)
  
  Xtrain = model.matrix(~ . -1, data = Xtrain_merge[, feature_cols, drop=F])
  Ytrain = Xtrain_merge$Rating
  return(list(Xtrain = Xtrain, Ytrain = Ytrain, 
              Xtrain_all = Xtrain_all, feature_cols = feature_cols))
}

output = preprocess(Xtrain, dests)
feature_cols = output$feature_cols
Xtrain_all = output$Xtrain_all
Xtrain = output$Xtrain
Ytrain = output$Ytrain

train_cols = colnames(Xtrain)
align_matrix = function(X_new, train_cols) {
  missing_cols = setdiff(train_cols, colnames(X_new))
  if (length(missing_cols) > 0) {
    add = matrix(0, nrow = nrow(X_new), ncol = length(missing_cols))
    colnames(add) = missing_cols
    X_new = cbind(X_new, add)
  }
  return(X_new[, train_cols, drop=F])
}

################################################################################
### 2. Train All 7 Models Parallel
train_all_rating = function(Xtrain, Ytrain, k_knn = 17, k_kmc = 3,
                            ntree_rf = 1000, seed = 1234) {
  
  # 1) KNN regression
  knn_model = list(X = Xtrain, Y = Ytrain, k = k_knn)
  
  # 2) DT regression
  dt_df = data.frame(Rating = Ytrain, Xtrain)
  dt_model = rpart(Rating ~ ., data = dt_df, method = "anova",
                   control = rpart.control(cp = 0.001, minsplit = 10))
  
  # 3) RF regression
  rf_model = randomForest(x = Xtrain, y = Ytrain,
                          ntree = ntree_rf,
                          mtry = floor(sqrt(ncol(Xtrain))))
  
  # 4) SVM regression
  svm_model = e1071::svm(x = Xtrain, y = Ytrain, 
                         type  = "eps-regression", kernel = "radial")
  
  # 5) KMC
  kmc_model_raw = kmc(Xtrain, k = k_kmc, init = "Forgy", iter = "Lloyd", seed = seed)
  mean_rate = tapply(Ytrain, kmc_model_raw$cids, mean)
  kmc_model = list(kmc = kmc_model_raw, mean_rate = mean_rate)
  
  # 6) LightGBM regression
  lgbm_train = lgb.Dataset(data = Xtrain, label = Ytrain)
  lgbm_params = list(objective        = "regression",
                     metric           = "rmse",
                     learning_rate    = 0.05,
                     num_leaves       = 31,
                     feature_fraction = 0.8,
                     bagging_fraction = 0.8,
                     bagging_freq     = 1)
  
  lgbm_model = lgb.train(params  = lgbm_params,
                         data    = lgbm_train,
                         nrounds = 400,
                         verbose = -1)
  
  # 7) XGBoost regression
  dtrain = xgb.DMatrix(data = Xtrain, label = Ytrain)
  params = list(objective        = "reg:squarederror",
                eval_metric      = "rmse",
                max_depth        = 4,
                eta              = 0.05,
                subsample        = 0.8,
                colsample_bytree = 0.8)
  xgb_model = xgb.train(params = params, data = dtrain, nrounds = 400, verbose = 0)
  
  return(list("knn"   = knn_model,
              "dt"    = dt_model,
              "rf"    = rf_model,
              "svm"   = svm_model,
              "kmc"   = kmc_model,
              "lgbm"  = lgbm_model,
              "xgb"   = xgb_model))
}

models = train_all_rating(Xtrain, Ytrain)

################################################################################
### 3. Per-model rating predictors
# 1) KNN average neighbor ratings
knn_rating = function(model, Xnew) {
  n_new = nrow(Xnew)
  ratings = numeric(n_new)
  
  for (i in 1:n_new) {
    KNN = knn(model$X, model$Y, Xnew[i, ], k = model$k, type = 2, ztype = "z-score")
    labs = as.numeric(KNN$k_nearest_labels)
    ratings[i] = mean(labs)
  }
  return(ratings)
}

# 2) DT regression
dt_rating = function(model, Xnew) {
  as.numeric(predict(model, newdata = as.data.frame(Xnew)))
}

# 3) RF regression
rf_rating = function(model, Xnew) {
  as.numeric(predict(model, newdata = Xnew))
}

# 4) SVM regression
svm_rating = function(model, Xnew) {
  as.numeric(predict(model, newdata = Xnew))
}

# 5) KMC cluster mean rating
kmc_rating = function(model, Xnew) {
  km = model$kmc
  mr = model$mean_rate
  n_new = nrow(Xnew)
  ratings = numeric(n_new)
  
  for (i in 1:n_new) {
    cid = km$nearest_cids(Xnew[i, ])
    v = mr[as.character(cid)]
    ratings[i] = ifelse(is.na(v), mean(mr, na.rm=T), v)
  }
  return(ratings)
}

# 6) LightGBM regression
lgbm_rating = function(model, Xnew) {
  as.numeric(predict(model, Xnew))
}

# 7) XGB regression
xgb_rating = function(model, Xnew) {
  as.numeric(predict(model, xgb.DMatrix(Xnew)))
}

################################################################################
### Meta-Learner for Stacking by using MLR
base_models = c("knn", "dt", "rf", "svm", "kmc", "lgbm", "xgb")
M = length(base_models)

build_meta = function(Xtrain, Ytrain, Kfolds = 5, seed = 1234) {
  set.seed(seed)
  n = nrow(Xtrain)
  folds = sample(rep(1:Kfolds, length.out = n))
  
  P = matrix(NA, nrow = n, ncol = M)
  colnames(P) = base_models
  
  for (k in 1:Kfolds) {
    idx_tr  = which(folds != k)
    idx_val = which(folds == k)
    
    model_k = train_all_rating(Xtrain[idx_tr, , drop=F], Ytrain[idx_tr], seed = seed)
    
    P[idx_val, "knn"] = knn_rating(model_k$knn, Xtrain[idx_val, , drop=F])
    P[idx_val, "dt"] = dt_rating(model_k$dt, Xtrain[idx_val, , drop=F])
    P[idx_val, "rf"] = rf_rating(model_k$rf, Xtrain[idx_val, , drop=F])
    P[idx_val, "svm"] = svm_rating(model_k$svm, Xtrain[idx_val, , drop=F])
    P[idx_val, "kmc"] = kmc_rating(model_k$kmc, Xtrain[idx_val, , drop=F])
    P[idx_val, "lgbm"] = lgbm_rating(model_k$lgbm, Xtrain[idx_val, , drop=F])
    P[idx_val, "xgb"] = xgb_rating(model_k$xgb, Xtrain[idx_val, , drop=F])
  }
  
  return(list(P = P, Y = Ytrain))
}

meta_data = build_meta(Xtrain, Ytrain, Kfolds = 5, seed = 1234)
meta_df = data.frame(Rating = meta_data$Y, meta_data$P)
head(meta_df, 20)

stack = lm(Rating ~ . -1, data = meta_df)
stack$coef

################################################################################
### 5. Models Ensemble: Stacking
stack_predict_rating = function(models, stack_model, Xnew) {
  comps = data.frame(knn = knn_rating(models$knn, Xnew),
                     dt  = dt_rating(models$dt, Xnew),
                     rf  = rf_rating(models$rf, Xnew),
                     svm = svm_rating(models$svm, Xnew),
                     kmc = kmc_rating(models$kmc, Xnew),
                     lgbm  = lgbm_rating(models$lgbm, Xnew),
                     xgb = xgb_rating(models$xgb, Xnew))
  
  return(as.numeric(predict(stack_model, newdata = comps)))
}

################################################################################
### Case I: No user info
top5_no_info = function(Xtrain_all, dests, topk = 5) {
  avg = aggregate(Rating ~ LocationID, data = Xtrain_all, FUN = mean)
  avg = avg[order(-avg$Rating),]
  top = head(avg, topk)
  result = merge(top, dests, by = "LocationID")
  return(result[order(-result$Rating),])
}

top5_no_info(Xtrain_all, dests, topk = 5)

################################################################################
### Case II: Have user info
prepare_Xnew = function(Xnew, dests, models, stack) {
  Xnew$id = seq_len(nrow(Xnew))
  grid0 = merge(Xnew, dests, by = NULL)
  
  prepare_grid = function(grid) {
    Xg = model.matrix(~ . -1, data = grid[, feature_cols, drop=F])
    Xg = align_matrix(Xg, train_cols)
    list(grid = grid, Xg = Xg)
  }
  
  prep = prepare_grid(grid0)
  grid = prep$grid
  Xg = prep$Xg
  
  grid$PredRating = stack_predict_rating(models, stack, Xg)
  out = split(grid, grid$id)
  return(out)
}

recommend_topk_rating = function(Xnew, dests, topk = 5, models, stack, 
                                 climate_choice = NULL, city_choice = NULL, temp_choice = NULL) {
  out = prepare_Xnew(Xnew, dests, models, stack)
  result = lapply(out, function(df) {
    if (!is.null(climate_choice)) {
      df = df[which(df$ClimateZone == climate_choice),]
    }
    if (!is.null(city_choice)) {
      df = df[which(df$CityType == city_choice),]
    }
    if (!is.null(temp_choice)) {
      reg = gregexpr("-?\\d+\\.?\\d*", temp_choice)
      nums = as.numeric(unlist(regmatches(temp_choice, reg)))
      df = df[which(df$AvgTemperature_C > nums[1] & 
                    df$AvgTemperature_C <= nums[2]),]
    }
    df = df[order(df$PredRating, decreasing=T), , drop=F]
    return(head(data.frame(LocationID = df$LocationID, PredRating = df$PredRating), topk))
  })
  return(result)
}

top5_have_info = function(Xnew, dests, topk = 5, models, stack,
                          climate_choice = NULL, city_choice = NULL, temp_choice = NULL) {
  model = recommend_topk_rating(Xnew, dests, topk, models, stack,
                                climate_choice, city_choice, temp_choice)
  
  loc_list = vector("list", length = nrow(Xnew))
  for (i in seq_len(nrow(Xnew))) {
    top5 = model[[i]]
    merged  = merge(top5, dests, by = "LocationID")
    loc_list[[i]]  = merged[order(merged$PredRating, decreasing=T), ]
  }
  
  user_list <- split(Xnew, seq_len(nrow(Xnew)))
  result <- Map(function(user, dests) {
      list(user = user, dests = dests)
    }, user_list, loc_list)
  
  return(result)
}

# Example 1 (recommend based on test set)
new_users = Xtest[1:3,]
top5_have_info(new_users, dests, topk = 5, models, stack)

# Example 2 (interaction to get user info)
top5_ask_info = function(dests, topk = 5, models, stack) {
  age = readline("Please enter your age: ")
  parent = readline("Please enter your parental education years: ")
  ceo = readline("Please enter your CEO annual salary: ")
  budget = readline("Please enter your budget: ")
  duration = readline("Please enter your ideal trip duration days: ")
  lat = readline("Please enter your current latitude: ")
  long = readline("Please enter your current longitude: ")
  
  climate = menu(c("Cold", "Mixed", "Temperate", "Warm"), title = "Please choose one:")
  climate = c("Cold", "Mixed", "Temperate", "Warm")[climate]
  
  style = menu(c("Adventure", "Cultural", "Luxury", "Party", "Relaxation"), title = "Please choose one:")
  style = c("Adventure", "Cultural", "Luxury", "Party", "Relaxation")[style]
  
  compan = menu(c("Couples", "Family", "Friends", "Solo", "Work"), title = "Please choose one:")
  compan = c("Couples", "Family", "Friends", "Solo", "Work")[compan]
  
  color = menu(c("Black", "Blue", "Green", "Orange", "Purple", "Red", "White", "Yellow"), title = "Please choose one:")
  color = c("Black", "Blue", "Green", "Orange", "Purple", "Red", "White", "Yellow")[color]
  
  sport = menu(c("Losing", "Neutral", "Winning"), title = "Please choose one:")
  sport = c("Losing", "Neutral", "Winning")[sport]
  
  new_users = data.frame("Age" = as.numeric(age), "ParentalEducationYears" = as.numeric(parent), 
                         "CEOAnnualSalary" = as.numeric(ceo), "Budget" = as.numeric(budget), 
                         "IdealTripDurationDays" = as.numeric(duration), "CurrentLatitude" = as.numeric(lat), 
                         "CurrentLongitude" = as.numeric(long), "ClimatePreference" = climate, 
                         "TravelStylePreference" = style, "TravelCompanionPreference" = compan, 
                         "FavoriteColor" = color, "LocalSportsTeamRecord" = sport)
  
  new_users$ClimatePreference = factor(new_users$ClimatePreference, levels = levels(users$ClimatePreference))
  new_users$TravelStylePreference = factor(new_users$TravelStylePreference, levels = levels(users$TravelStylePreference))
  new_users$TravelCompanionPreference = factor(new_users$TravelCompanionPreference, levels = levels(users$TravelCompanionPreference))
  new_users$FavoriteColor = factor(new_users$FavoriteColor, levels = levels(users$FavoriteColor))
  new_users$LocalSportsTeamRecord = factor(new_users$LocalSportsTeamRecord, levels = levels(users$LocalSportsTeamRecord))
  
  return(top5_have_info(new_users, dests, topk, models, stack))
}

top5_ask_info(dests, topk = 5, models, stack)

################################################################################
### Case III: Have user info (can ask follow up questions about destination info)
top5_followup_info = function(new_users, dests, topk = 5, models, stack) {
  climate_options = menu(c("Dry", "Temperate", "Continental", "Tropical", "Polar", "I'm fine with all"), 
                         title = "Please choose one: ")
  if (climate_options == 6) {
    climate_choice = NULL
  } else {
    climate_choice = c("Dry", "Temperate", "Continental", "Tropical", "Polar")[climate_options]
  }
  
  climate_choice = ifelse(climate_options == 6, NULL, 
                          c("Dry", "Temperate", "Continental", "Tropical", "Polar")[climate_options])
  
  city_options = menu(c("Cultural", "Resort", "Beach", "Urban", "Natural", "Adventure", "I'm fine with all"), 
                      title = "Please choose one: ")
  if (city_options == 7) {
    city_choice = NULL
  } else {
    city_choice = c("Cultural", "Resort", "Beach", "Urban", "Natural", "Adventure")[city_options]
  }
  
  temp_options = menu(c("-10 < temp <= 0", "0 < temp <= 10", "10 < temp <= 20", "20 < temp <= 33", "I'm fine with all"), 
                      title = "Please choose one: ")
  if (temp_options == 5) {
    temp_choice = NULL
  } else {
    temp_choice = c("-10 < temp <= 0", "0 < temp <= 10", "10 < temp <= 20", "20 < temp <= 33")[temp_options]
  }
  
  return(top5_have_info(new_users, dests, topk, models, stack, climate_choice, city_choice, temp_choice))
}

new_users = Xtest[1,]
top5_followup_info(new_users, dests, topk = 5, models, stack)

################################################################################
### Case IV: Partial Predictors (have same predictors as another team)
X1 = users$Budget
X2 = users$IdealTripDurationDays
X3 = users$CurrentLatitude
X4 = users$CurrentLongitude
X5 = users$ClimatePreference
X6 = users$PastTripsWithRatings
users4 = data.frame("Budget" = X1, 
                   "IdealTripDurationDays" = X2, 
                   "CurrentLatitude" = X3, 
                   "CurrentLongitude" = X4, 
                   "ClimatePreference" = X5,
                   "PastTripsWithRatings" = X6)
users4$ClimatePreference = factor(users4$ClimatePreference)

Xtrain4 = users[itrain,]
Xtest4 = users[itest,]
output4 = preprocess(Xtrain4, dests)
feature_cols4 = output4$feature_cols
Xtrain4 = output$Xtrain
Ytrain4 = output$Ytrain
train_cols4 = colnames(Xtrain4)

models4 = train_all_rating(Xtrain4, Ytrain4)
meta_data4 = build_meta(Xtrain4, Ytrain4, Kfolds = 5, seed = 1234)
meta_df4 = data.frame(Rating = meta_data4$Y, meta_data4$P)
stack4 = lm(Rating ~ . -1, data = meta_df4)

new_users = Xtest4[1,]
top5_have_info(new_users, dests, topk = 5, models4, stack)















### The following code is the sidenote for hyperparameter tuning
################################################################################
### Hyperparameter Tuning for KNN
rmse = function(y, yhat) {
  sqrt(mean((y - yhat)^2))
}

tune_knn_k = function(X, y, k_grid = seq(3, 31, 2), Kfolds = 5, seed = 1234) {
  set.seed(seed)
  n = nrow(X)
  folds = sample(rep(1:Kfolds, length.out = n))
  results = data.frame(k = k_grid, RMSE = NA_real_)
  
  for (i in seq_along(k_grid)) {
    k_val = k_grid[i]
    rmse_vec = numeric(Kfolds)
    
    for (f in 1:Kfolds) {
      idx_tr  = which(folds != f)
      idx_val = which(folds == f)
      
      model_fold = list(X = X[idx_tr, , drop=F],
                        Y = y[idx_tr],
                        k = k_val)
      
      preds = knn_rating(model_fold, X[idx_val, , drop=F])
      rmse_vec[f] = rmse(y[idx_val], preds)
    }
    
    results$RMSE[i] = mean(rmse_vec)
  }
  
  best_idx = which.min(results$RMSE)
  return(list(best_k = results$k[best_idx], cv_results = results))
}

knn_tuning = tune_knn_k(Xtrain, Ytrain)
knn_tuning$best_k
knn_tuning$cv_results


################################################################################
### Hyperparameter Tuning for KMC
tune_kmc_k = function(X, y, k_grid = c(3, 5, 7, 10, 15), 
                      Kfolds = 5, seed = 1234) {
  set.seed(seed)
  n = nrow(X)
  folds = sample(rep(1:Kfolds, length.out = n))
  results = data.frame(k = k_grid, RMSE = NA_real_)
  
  for (i in seq_along(k_grid)) {
    k_val = k_grid[i]
    rmse_vec = numeric(Kfolds)
    
    for (f in 1:Kfolds) {
      idx_tr  = which(folds != f)
      idx_val = which(folds == f)
      
      kmc_model_raw = kmc(X[idx_tr, , drop=F], k = k_val,
                          init = "Forgy", iter = "Lloyd", seed = seed)
      mean_rate = tapply(y[idx_tr], kmc_model_raw$cids, mean)
      kmc_model = list(kmc = kmc_model_raw, mean_rate = mean_rate)
      
      preds = kmc_rating(kmc_model, X[idx_val, , drop=F])
      rmse_vec[f] = rmse(y[idx_val], preds)
    }
    
    results$RMSE[i] = mean(rmse_vec)
  }
  
  best_idx = which.min(results$RMSE)
  return(list(best_k = results$k[best_idx], cv_results = results))
}

kmc_tuning = tune_kmc_k(Xtrain, Ytrain)
kmc_tuning$best_k
kmc_tuning$cv_results

################################################################################
### Hyperparameter Tuning for RF
tune_rf_ntree = function(X, y, ntree_grid = c(500, 1000, 1500, 2000),
                         Kfolds = 5, seed = 1234) {
  set.seed(seed)
  n = nrow(X)
  folds = sample(rep(1:Kfolds, length.out = n))
  results = data.frame(ntree = ntree_grid, RMSE = NA_real_)
  
  for (i in seq_along(ntree_grid)) {
    ntree_val = ntree_grid[i]
    rmse_vec = numeric(Kfolds)
    
    for (f in 1:Kfolds) {
      idx_tr  = which(folds != f)
      idx_val = which(folds == f)
      
      rf_model = randomForest(x = X[idx_tr, , drop=F],
                              y = y[idx_tr],
                              ntree = ntree_val,
                              mtry = floor(sqrt(ncol(X))))
      
      preds = predict(rf_model, newdata = X[idx_val, , drop=F])
      rmse_vec[f] = rmse(y[idx_val], preds)
    }
    
    results$RMSE[i] = mean(rmse_vec)
  }
  
  best_idx = which.min(results$RMSE)
  return(list(best_ntree = results$ntree[best_idx], cv_results = results))
}

rf_tuning = tune_rf_ntree(Xtrain, Ytrain)
rf_tuning$best_ntree
rf_tuning$cv_results
