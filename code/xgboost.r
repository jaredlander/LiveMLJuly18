library(useful)
library(xgboost)
library(coefplot)
library(magrittr)
library(dygraphs)

land_train <- readr::read_csv('data/manhattan_Train.csv')
land_test <- readRDS('data/manhattan_Test.rds')
land_val <- readRDS('data/manhattan_Validate.rds')

table(land_train$HistoricDistrict)

histFormula <- HistoricDistrict ~ FireService + 
    ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + 
    ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + 
    NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + 
    BldgDepth + LotType + Landmark + BuiltFAR +
    Built + TotalValue - 1

landX_train <- build.x(histFormula, data=land_train, contrasts=FALSE, sparse=TRUE)
landY_train <- build.y(histFormula, data=land_train) %>% as.factor() %>% as.integer() - 1
head(landY_train, n=20)

landX_val <- build.x(histFormula, data=land_val, contrasts=FALSE, sparse=TRUE)
landY_val <- build.y(histFormula, data=land_val) %>% as.factor() %>% as.integer() - 1

landX_test <- build.x(histFormula, data=land_test, contrasts=FALSE, sparse=TRUE)
landY_test <- build.y(histFormula, data=land_test) %>% as.factor() %>% as.integer() - 1

xgTrain <- xgb.DMatrix(data=landX_train, label=landY_train)
xgVal <- xgb.DMatrix(data=landX_val, label=landY_val)

xg1 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1
)
xg1

xgb.plot.multi.trees(xg1, feature_names=colnames(landX_train))

xg2 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1,
    watchlist=list(train=xgTrain)
)

xg3 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=100,
    watchlist=list(train=xgTrain),
    print_every_n=1
)

xg4 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=300,
    watchlist=list(train=xgTrain),
    print_every_n=1
)

xg5 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=500,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1
)

xg5$evaluation_log
dygraph(xg5$evaluation_log)

xg6 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70
)
dygraph(xg6$evaluation_log)
xg6$best_iteration
xg6$best_score

xgb.plot.importance(
    xgb.importance(
        xg6, feature_names=colnames(landX_train)
    )
)


xg7 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gblinear',
    nrounds=1000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70
)

coefplot(xg7, sort='magnitude')

xg8 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gblinear',
    nrounds=1000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    alpha=1000, lambda=1250
)

dygraph(xg8$evaluation_log)


xg9 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8
)

xg10 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=3
)

xg11 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=2500,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1
)

xg12 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=2500,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.5
)

xg13 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=2500,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.5, colsample_bytree=0.5
)

xg14 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=1,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.5, colsample_bytree=0.5,
    num_parallel_tree=100
)


xg15 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=100,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=1,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    subsample=0.5, colsample_bytree=0.5,
    num_parallel_tree=20
)

xg16 <- xgb.train(
    data=xgTrain,
    objective='binary:logistic',
    eval_metric='logloss',
    booster='gbtree',
    nrounds=2000,
    watchlist=list(train=xgTrain, validate=xgVal),
    print_every_n=10,
    early_stopping_rounds=70,
    max_depth=8, eta=0.1,
    nthread=2
)

?xgb.train

xgPreds16 <- predict(xg16, newdata=landX_test, outputmargin=FALSE)
xgPreds16 %>% head(n=20) %>% round(2)
