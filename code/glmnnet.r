library(glmnet)
library(useful)
library(coefplot)
library(magrittr)

land_train <- readr::read_csv('data/manhattan_Train.csv')
land_test <- readRDS('data/manhattan_Test.rds')

View(land_train)

valueFormula <- TotalValue ~ FireService + 
    ZoneDist1 + ZoneDist2 + Class + LandUse + 
    OwnerType + LotArea + BldgArea + ComArea + 
    ResArea + OfficeArea + RetailArea + 
    GarageArea + FactryArea + NumBldgs + 
    NumFloors + UnitsRes + UnitsTotal + 
    LotFront + LotDepth + BldgFront + 
    BldgDepth + LotType + Landmark + BuiltFAR +
    Built + HistoricDistrict - 1

valueFormula
class(valueFormula)

value1 <- lm(valueFormula, data=land_train)
coefplot(value1, sort='magnitude')

summary(value1)

landX_train <- build.x(valueFormula, data=land_train, contrasts=FALSE, sparse=TRUE)
landY_train <- build.y(valueFormula, data=land_train)

value2 <- glmnet(x=landX_train, y=landY_train, family='gaussian')
coefpath(value2)
