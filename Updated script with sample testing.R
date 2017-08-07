library(corrplot)
library(ggplot2)
library(stringr)
library(e1071)
library(Matrix)
library(plyr)
library(dplyr)
library(xgboost)
library(randomForest)
library(caret)
library(Metrics)
library(scales)
library(glmnet)
library(NLP)

data <- read.csv("train.csv")

dim(data)
str(data)
colnames(data)

missing <- colSums(sapply(data, is.na))
missing
 #          Id    MSSubClass      MSZoning   LotFrontage       LotArea        Street         Alley 
 #           0             0             0           259             0             0          1369 
 #    LotShape   LandContour     Utilities     LotConfig     LandSlope  Neighborhood    Condition1 
 #           0             0             0             0             0             0             0 
 #  Condition2      BldgType    HouseStyle   OverallQual   OverallCond     YearBuilt  YearRemodAdd 
 #           0             0             0             0             0             0             0 
 #   RoofStyle      RoofMatl   Exterior1st   Exterior2nd    MasVnrType    MasVnrArea     ExterQual 
 #           0             0             0             0             8             8             0 
 #   ExterCond    Foundation      BsmtQual      BsmtCond  BsmtExposure  BsmtFinType1    BsmtFinSF1 
 #           0             0            37            37            38            37             0 
 #BsmtFinType2    BsmtFinSF2     BsmtUnfSF   TotalBsmtSF       Heating     HeatingQC    CentralAir 
 #          38             0             0             0             0             0             0 
 #  Electrical     X1stFlrSF     X2ndFlrSF  LowQualFinSF     GrLivArea  BsmtFullBath  BsmtHalfBath 
 #           1             0             0             0             0             0             0 
 #    FullBath      HalfBath  BedroomAbvGr  KitchenAbvGr   KitchenQual  TotRmsAbvGrd    Functional 
 #           0             0             0             0             0             0             0 
 #  Fireplaces   FireplaceQu    GarageType   GarageYrBlt  GarageFinish    GarageCars    GarageArea 
 #           0           690            81            81            81             0             0 
 #  GarageQual    GarageCond    PavedDrive    WoodDeckSF   OpenPorchSF EnclosedPorch    X3SsnPorch 
 #          81            81             0             0             0             0             0 
 # ScreenPorch      PoolArea        PoolQC         Fence   MiscFeature       MiscVal        MoSold 
 #          0             0          1453          1179          1406             0             0 
 #      YrSold      SaleType SaleCondition     SalePrice 
 #           0             0             0             0 



#Remove the missing electrical data tuple
data <- data[-1380,]

#Remove the other missing data attributes and ID attr:
data <- data[,-c(1,4,7,26,27,31,32,33,34,36,58,59,60,61,64,65,73,74,75)]

missing <- colSums(sapply(data, is.na))
missing
#MSSubClass      MSZoning       LotArea        Street      LotShape   LandContour     Utilities     LotConfig 
#0             0             0             0             0             0             0             0 
#LandSlope  Neighborhood    Condition1    Condition2      BldgType    HouseStyle   OverallQual   OverallCond 
#0             0             0             0             0             0             0             0 
#YearBuilt  YearRemodAdd     RoofStyle      RoofMatl   Exterior1st   Exterior2nd     ExterQual     ExterCond 
#0             0             0             0             0             0             0             0 
#Foundation    BsmtFinSF1    BsmtFinSF2     BsmtUnfSF   TotalBsmtSF       Heating     HeatingQC    CentralAir 
#0             0             0             0             0             0             0             0 
#Electrical     X1stFlrSF     X2ndFlrSF  LowQualFinSF     GrLivArea  BsmtFullBath  BsmtHalfBath      FullBath 
#0             0             0             0             0             0             0             0 
#HalfBath  BedroomAbvGr  KitchenAbvGr   KitchenQual  TotRmsAbvGrd    Functional    Fireplaces    GarageCars 
#0             0             0             0             0             0             0             0 
#GarageArea    PavedDrive    WoodDeckSF   OpenPorchSF EnclosedPorch    X3SsnPorch   ScreenPorch      PoolArea 
#0             0             0             0             0             0             0             0 
#MiscVal        MoSold        YrSold      SaleType SaleCondition     SalePrice 
#0             0             0             0             0             0 

#Function to plot categorical attributes 
plot.categoric = function(cols, data){
  for (col in cols) {
    order.cols = names(sort(table(data[,col]), decreasing = TRUE))
    
    num.plot = qplot(data[,col]) +
      geom_bar(fill = 'blue') +
      geom_text(aes(label = ..count..), stat='count', vjust=-0.5) +
      theme_minimal() +
      scale_y_continuous(limits = c(0,max(table(data[,col]))*1.1)) +
      scale_x_discrete(limits = order.cols) +
      xlab(col) +
      theme(axis.text.x = element_text(angle = 30, size=12))
    
    print(num.plot)
  }
}

#Splitting the numeric and non-numeric attributes up for evaluation
numeric_att <- data[which(sapply(data, is.numeric))]
other_cat <- data[-which(sapply(data, is.numeric))]

###Evaluating Numerical Attributes###

colnames(numeric_att)
#[1] "MSSubClass"    "LotArea"       "OverallQual"   "OverallCond"   "YearBuilt"     "YearRemodAdd"  "BsmtFinSF1"    "BsmtFinSF2"   
#[9] "BsmtUnfSF"     "TotalBsmtSF"   "X1stFlrSF"     "X2ndFlrSF"     "LowQualFinSF"  "GrLivArea"     "BsmtFullBath"  "BsmtHalfBath" 
#[17] "FullBath"      "HalfBath"      "BedroomAbvGr"  "KitchenAbvGr"  "TotRmsAbvGrd"  "Fireplaces"    "GarageCars"    "GarageArea"   
#[25] "WoodDeckSF"    "OpenPorchSF"   "EnclosedPorch" "X3SsnPorch"    "ScreenPorch"   "PoolArea"      "MiscVal"       "MoSold"       
#[33] "YrSold"        "SalePrice"  


#Conducting a correlation test on each numeric attribute, with the output below:
for (i in 1:length(numeric_att)) {
  print(colnames(numeric_att[i]))
  print(cor.test(data$SalePrice, numeric_att[,i]))
}

#[1] "MSSubClass"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -3.2266, df = 1457, p-value = 0.001281
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
-0.13496676 -0.03305328
sample estimates:
cor 
-0.08423029 
"
#[1] "LotArea"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 10.441, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.2154345 0.3109470
sample estimates:
  cor 
0.2638374 
"
#[1] "OverallQual"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 49.361, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.7710516 0.8095239
sample estimates:
cor 
0.7910687 
"
#[1] "OverallCond"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -2.9835, df = 1457, p-value = 0.002897
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.12872881 -0.02671064
sample estimates:
  cor 
-0.07792371 
"
#[1] "YearBuilt"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 23.439, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.4849769 0.5595662
sample estimates:
cor 
0.5232731 
"
#[1] "YearRemodAdd"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 22.478, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.4683056 0.5445688
sample estimates:
  cor 
0.5074302 
"
#[1] "BsmtFinSF1"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 15.993, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.3418968 0.4292435
sample estimates:
cor 
0.3864363 
"
#[1] "BsmtFinSF2"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -0.43563, df = 1457, p-value = 0.6632
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.06269516  0.03993128
sample estimates:
  cor 
-0.01141199 
"
#[1] "BsmtUnfSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 8.3805, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.1649411 0.2628727
sample estimates:
cor 
0.2144458 
"
#[1] "TotalBsmtSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 29.686, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.5808862 0.6449067
sample estimates:
  cor 
0.613905 
"
#[1] "X1stFlrSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 29.077, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.5724501 0.6374638
sample estimates:
cor 
0.6059679 
"
#[1] "X2ndFlrSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 12.868, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.2726136 0.3648031
sample estimates:
  cor 
0.3194641 
"
#[1] "LowQualFinSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -0.97827, df = 1457, p-value = 0.3281
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
-0.07683929  0.02573329
sample estimates:
cor 
-0.02562044 
"
#[1] "GrLivArea"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 38.334, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.6821032 0.7332712
sample estimates:
  cor 
0.7086176 
"
#[1] "BsmtFullBath"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 8.9004, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.1778344 0.2751946
sample estimates:
cor 
0.2270818 
"
#[1] "BsmtHalfBath"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -0.64413, df = 1457, p-value = 0.5196
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.06813347  0.03447716
sample estimates:
  cor 
-0.01687258 
"
#[1] "FullBath"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 25.86, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.5246628 0.5950718
sample estimates:
cor 
0.5608806 
"
#[1] "HalfBath"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 11.323, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.2365319 0.3308900
sample estimates:
  cor 
0.2843996 
"
#[1] "BedroomAbvGr"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 6.5145, df = 1457, p-value = 1.002e-10
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.1179336 0.2176758
sample estimates:
cor 
0.1682353 
"
#[1] "KitchenAbvGr"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -5.2374, df = 1457, p-value = 1.868e-07
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.18595789 -0.08520983
sample estimates:
  cor 
-0.1359353 
"
#[1] "TotRmsAbvGrd"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 24.094, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.4960474 0.5694982
sample estimates:
cor 
0.5337789 
"
#[1] "Fireplaces"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 20.157, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.4258536 0.5061580
sample estimates:
  cor 
0.466968 
"
#[1] "GarageCars"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 31.833, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.6091760 0.6697779
sample estimates:
cor 
0.6404729 
"
#[1] "GarageArea"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 30.435, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.5910118 0.6538243
sample estimates:
  cor 
0.6234229 
"
#[1] "WoodDeckSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 13.091, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.2777262 0.3695887
sample estimates:
cor 
0.3244222 
"
#[1] "OpenPorchSF"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 12.706, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.2688695 0.3612953
sample estimates:
  cor 
0.3158314 
"
#[1] "EnclosedPorch"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -4.9509, df = 1457, p-value = 8.248e-07
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
-0.17876636 -0.07782034
sample estimates:
cor 
-0.1286265 
"
#[1] "X3SsnPorch"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 1.703, df = 1457, p-value = 0.08878
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.006764528  0.095671883
sample estimates:
  cor 
0.04457083 
"
#[1] "ScreenPorch"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 4.2796, df = 1457, p-value = 1.995e-05
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
0.06044489 0.16181378
sample estimates:
cor 
0.1114192 
"
#[1] "PoolArea"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 3.542, df = 1457, p-value = 0.0004096
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  0.04127246 0.14303828
sample estimates:
  cor 
0.09239665 
"
#[1] "MiscVal"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -0.8094, df = 1457, p-value = 0.4184
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
-0.07244107  0.03015269
sample estimates:
cor 
-0.0212 
"
#[1] "MoSold"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = 1.7723, df = 1457, p-value = 0.07656
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  -0.004951827  0.097467741
sample estimates:
  cor 
0.04637985 
"
#[1] "YrSold"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = -1.1039, df = 1457, p-value = 0.2698
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
-0.08010803  0.02244620
sample estimates:
cor 
-0.02890698 
"
#[1] "SalePrice"
"
Pearson's product-moment correlation

data:  data$SalePrice and numeric_att[, i]
t = Inf, df = 1457, p-value < 2.2e-16
alternative hypothesis: true correlation is not equal to 0
95 percent confidence interval:
  1 1
sample estimates:
  cor 
1"

remove <- list("")
#Removing the low correlated numeric attributes ( < 50% correlation)
for (i in 1:length(data)) {
  if (is.numeric(data[,i])) {
    if (cor(data$SalePrice, data[,i]) < .5) {
      remove <- append(remove, colnames(data[i]), after = length(remove))
    }
  }
}

remove
#[1] "MSSubClass", "LotArea", "OverallCond", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "X2ndFlrSF", "LowQualFinSF", "BsmtFullBath", "BsmtHalfBath", "HalfBath",
# "BedroomAbvGr", "KitchenAbvGr", "Fireplaces", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "X3SsnPorch", "ScreenPorch", "PoolArea", "MiscVal", "MoSold", "YrSold"

data <- data[-which(colnames(data) %in% remove)]


#Updating the numeric attributes list to reflect changes
numeric_att <- data[which(sapply(data, is.numeric))]



###Evaluating Categorical Attributes###

#Removing the low variance non-numeric attributes (removed if 1 factor accounts for > 85% of all classifications within attribute)
remove <- c("Street", "LandContour",  "LandSlope", "Utilities", "Condition1", "Condition2", "RoofMatl", "Foundation", "Heating", "Electrical",
            "Functional", "PavedDrive", "SaleType")
data <- data[-which(colnames(data) %in% remove)]


#Updating categorical list:
other_cat <- data[-which(sapply(data, is.numeric))]

#Plotting the categorical list 
for (i in 1:length(other_cat)) {
  plot.categoric(colnames(other_cat[i]), other_cat)
}

#Evaluating the boxplot of all categorical data to determine if there is a relationship with SalePrice:
boxplot(data$SalePrice ~ data$MSZoning, ylab=("Sale Price"), xlab=("MS Zoning"))#Looks linear
boxplot(data$SalePrice ~ data$LotShape, ylab=("Sale Price"), xlab=("Lot Shape"))#None
boxplot(data$SalePrice ~ data$LotConfig, ylab=("Sale Price"), xlab=("Lot Config"))#None
boxplot(data$SalePrice ~ data$BldgType, ylab=("Sale Price"), xlab=("Bldg Type"))#Looks Linear
boxplot(data$SalePrice ~ data$HouseStyle, ylab=("Sale Price"), xlab=("House Style"))#Looks Linear
boxplot(data$SalePrice ~ data$RoofStyle, ylab=("Sale Price"), xlab=("Roof Style"))#None
boxplot(data$SalePrice ~ data$Exterior1st, ylab=("Sale Price"), xlab=("Exterior 1st"))#None
boxplot(data$SalePrice ~ data$Exterior2nd, ylab=("Sale Price"), xlab=("Exterior 2nd"))#None
boxplot(data$SalePrice ~ data$ExterQual, ylab=("Sale Price"), xlab=("Exterior Qual"))#Looks Linear
boxplot(data$SalePrice ~ data$ExterCond, ylab=("Sale Price"), xlab=("Exter Cond"))#Looks Linear
boxplot(data$SalePrice ~ data$HeatingQC, ylab=("Sale Price"), xlab=("Heating QC"))#Looks Linear
boxplot(data$SalePrice ~ data$CentralAir, ylab=("Sale Price"), xlab=("Central Air")) #Looks Linear
boxplot(data$SalePrice ~ data$KitchenQual, ylab=("Sale Price"), xlab=("Kitchen Qual"))#Looks Linear
boxplot(data$SalePrice ~ data$SaleCondition, ylab=("Sale Price"), xlab=("Sale Condition"))#Looks Linear

#Removing the categorical attributes that don't show a linear relationship with SalePrice
remove <- c("LotShape", "LotConfig", "RoofStyle", "Exterior1st", "Exterior2nd")
data <- data[-which(colnames(data) %in% remove)]


##Turning Non-Ordinal Categorization Attributes to Numerical##

#Plotting the median value of each neighborhood grouping:
data[,c('Neighborhood','SalePrice')] %>%
  group_by(Neighborhood) %>%
  summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
  arrange(median.price) %>%
  mutate(nhbr.sorted = factor(Neighborhood, levels=Neighborhood)) %>%
  ggplot(aes(x=nhbr.sorted, y=median.price)) +
  geom_point() +
  theme_minimal() +
  labs(x='Neighborhood', y='Median price') +
  theme(text = element_text(size=12))

#The above plot shows that there are 6 distinct levels in the median prices in the neighborhoods
#Creating numerical categories for each neighborhood setting:
nFactor1 <- c("MeadowV", "IDOTRR", "BrDale")
nFactor2 <- c("OldTown", "Edwards", "BrkSide")
nFactor3 <- c("Sawyer", "Blueste", "SWISU", "NAmes", "NPkVill", "Mitchel")
nFactor4 <- c("SawyerW", "Gilbert", "NWAmes", "Blmngtn", "CollgCr", "ClearCr", "Crawfor")
nFactor5 <- c("Veenker", "Somerst", "Timber")
nFactor6 <- c("StoneBr", "NoRidge", "NridgHt")

for (i in 1:length(data[,"Neighborhood"])) {
  if (data[i, "Neighborhood"] %in% nFactor1) {
    data$NHood[i] <- 1
  }
  if (data[i, "Neighborhood"] %in% nFactor2) {
    data$NHood[i] <- 2
  }
  if (data[i, "Neighborhood"] %in% nFactor3) {
    data$NHood[i] <- 3
  }
  if (data[i, "Neighborhood"] %in% nFactor4) {
    data$NHood[i] <- 4
  }
  if (data[i, "Neighborhood"] %in% nFactor5) {
    data$NHood[i] <- 5
  }
  if (data[i, "Neighborhood"] %in% nFactor6) {
    data$NHood[i] <- 6
  }
}
boxplot(data$SalePrice ~ data$NHood, xlab="Numeric Neighborhood Value", ylab = "Sale Price")

#MSZoning:
data[,c('MSZoning','SalePrice')] %>%
       group_by(MSZoning) %>%
       summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
       arrange(median.price) %>%
       mutate(nhbr.sorted = factor(MSZoning, levels=MSZoning)) %>%
       ggplot(aes(x=nhbr.sorted, y=median.price)) +
       geom_point() +
       theme_minimal() +
       labs(x='MSZoning', y='Median price') +
       theme(text = element_text(size=12))

#Based on the graph, 5 levels will be assigned values, based on their median sale price:
for (i in 1:length(data[,"MSZoning"])) {
  if (data[i, "MSZoning"] %in% "C (all)") {
    data$MSZone[i] <- 1
  }
  if (data[i, "MSZoning"] %in% "RM") {
    data$MSZone[i] <- 2
  }
  if (data[i, "MSZoning"] %in% "RH") {
    data$MSZone[i] <- 3
  }
  if (data[i, "MSZoning"] %in% "RL") {
    data$MSZone[i] <- 4
  }
  if (data[i, "MSZoning"] %in% "FV") {
    data$MSZone[i] <- 5
  }
}
boxplot(data$SalePrice ~ data$MSZone)

#BldgeType:
data[,c('BldgType','SalePrice')] %>%
       group_by(BldgType) %>%
       summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
       arrange(median.price) %>%
       mutate(nhbr.sorted = factor(BldgType, levels=BldgType)) %>%
       ggplot(aes(x=nhbr.sorted, y=median.price)) +
       geom_point() +
       theme_minimal() +
       labs(x='BldgType', y='Median price') +
       theme(text = element_text(size=12))

#Based on the graph, there are 3 categorical levels. Assigning them numerically based on price:
for (i in 1:length(data[,"BldgType"])) {
  if (as.String(data$BldgType[i]) == "2fmCon") {
    data$BType[i] <- 1
  }
  if (as.String(data$BldgType[i]) == "Duplex") {
    data$BType[i] <- 2
  }
  if (as.String(data$BldgType[i]) == "Twnhs") {
    data$BType[i] <- 2
  }
  if (as.String(data$BldgType[i]) == "1Fam") {
    data$BType[i] <- 3
  }
  if (as.String(data$BldgType[i]) == "TwnhsE") {
    data$BType[i] <- 3
  }
}
boxplot(data$SalePrice ~ data$BType)

#HouseStyle:
data[,c('HouseStyle','SalePrice')] %>%
  group_by(HouseStyle) %>%
  summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
  arrange(median.price) %>%
  mutate(nhbr.sorted = factor(HouseStyle, levels=HouseStyle)) %>%
  ggplot(aes(x=nhbr.sorted, y=median.price)) +
  geom_point() +
  theme_minimal() +
  labs(x='HouseStyle', y='Median price') +
  theme(text = element_text(size=12))

#Based on the graph, the factors will be split into 4 levels, based on median sale price:
for (i in 1:length(data[,"HouseStyle"])) {
  if (as.String(data$HouseStyle[i]) == "1.5Unf") {
    data$HStyle[i] <- 1
  }
  if (as.String(data$HouseStyle[i]) == "1.5Fin") {
    data$HStyle[i] <- 2
  }
  if (as.String(data$HouseStyle[i]) == "2.5Unf") {
    data$HStyle[i] <- 2
  }
  if (as.String(data$HouseStyle[i]) == "SFoyer") {
    data$HStyle[i] <- 2
  }
  if (as.String(data$HouseStyle[i]) == "1Story") {
    data$HStyle[i] <- 3
  }
  if (as.String(data$HouseStyle[i]) == "SLvl") {
    data$HStyle[i] <- 3
  }
  if (as.String(data$HouseStyle[i]) == "2Story") {
    data$HStyle[i] <- 4
  }
  if (as.String(data$HouseStyle[i]) == "2.5Fin") {
    data$HStyle[i] <- 4
  }
}

boxplot(data$SalePrice ~ data$HStyle)

#SaleCondition
data[,c('SaleCondition','SalePrice')] %>%
       group_by(SaleCondition) %>%
       summarise(median.price = median(SalePrice, na.rm = TRUE)) %>%
       arrange(median.price) %>%
       mutate(nhbr.sorted = factor(SaleCondition, levels=SaleCondition)) %>%
       ggplot(aes(x=nhbr.sorted, y=median.price)) +
       geom_point() +
       theme_minimal() +
       labs(x='SaleCondition', y='Median price') +
       theme(text = element_text(size=12))

#This one shows a mostly linear plot, with the exception of "Partial"--All will be assigned diff values, based on their median sale price
for (i in 1:length(data[,"SaleCondition"])) {
  if (as.String(data$SaleCondition[i]) == "AdjLand") {
    data$SCond[i] <- 1
  }
  if (as.String(data$SaleCondition[i]) == "Abnormal") {
    data$SCond[i] <- 2
  }
  if (as.String(data$SaleCondition[i]) == "Family") {
    data$SCond[i] <- 3
  }
  if (as.String(data$SaleCondition[i]) == "Alloca") {
    data$SCond[i] <- 4
  }
  if (as.String(data$SaleCondition[i]) == "Normal") {
    data$SCond[i] <- 5
  }
  if (as.String(data$SaleCondition[i]) == "Partial") {
    data$SCond[i] <- 6
  }
}

boxplot(data$SalePrice ~ data$SCond)


##Turning Ordinal Categorical Attributes into Numerical##

#These have been renamed in accordance with their ranking and based off the explanation of what the ratings mean
#Kitchen Qual:
for (i in 1:length(data[,"KitchenQual"])) {
  if (data[i, "KitchenQual"] %in% "Fa") {
    data$KQual[i] <- 1
  }
  if (data[i, "KitchenQual"] %in% "TA") {
    data$KQual[i] <- 2
  }
  if (data[i, "KitchenQual"] %in% "Gd") {
    data$KQual[i] <- 3
  }
  if (data[i, "KitchenQual"] %in% "Ex") {
    data$KQual[i] <- 4
  }
}

boxplot(data$SalePrice ~ data$KQual)


#ExterQual:
for (i in 1:length(data[,"ExterQual"])) {
  if (data[i, "ExterQual"] %in% "Fa") {
    data$ExQual[i] <- 1
  }
  if (data[i, "ExterQual"] %in% "TA") {
    data$ExQual[i] <- 2
  }
  if (data[i, "ExterQual"] %in% "Gd") {
    data$ExQual[i] <- 3
  }
  if (data[i, "ExterQual"] %in% "Ex") {
    data$ExQual[i] <- 4
  }
}

boxplot(data$SalePrice ~ data$ExQual)

#CentralAir
for (i in 1:length(data[,"CentralAir"])) {
  if (data[i, "CentralAir"] %in% "N") {
    data$Cair[i] <- 0
  }
  if (data[i, "CentralAir"] %in% "Y") {
    data$Cair[i] <- 1
  }
}

boxplot(data$SalePrice ~ data$Cair)

#HeatingQC
for (i in 1:length(data[,"HeatingQC"])) {
  if (data[i, "HeatingQC"] %in% "Po") {
    data$HQC[i] <- 1
  }
  if (data[i, "HeatingQC"] %in% "Fa") {
    data$HQC[i] <- 2
  }
  if (data[i, "HeatingQC"] %in% "TA") {
    data$HQC[i] <- 3
  }
  if (data[i, "HeatingQC"] %in% "Gd") {
    data$HQC[i] <- 4
  }
  if (data[i, "HeatingQC"] %in% "Ex") {
    data$HQC[i] <- 5
  }
}

boxplot(data$SalePrice ~ data$HQC)

#ExterCond
for (i in 1:length(data[,"ExterCond"])) {
  if (data[i, "ExterCond"] %in% "Po") {
    data$ECond[i] <- 1
  }
  if (data[i, "ExterCond"] %in% "Fa") {
    data$ECond[i] <- 2
  }
  if (data[i, "ExterCond"] %in% "TA") {
    data$ECond[i] <- 3
  }
  if (data[i, "ExterCond"] %in% "Gd") {
    data$ECond[i] <- 4
  }
  if (data[i, "ExterCond"] %in% "Ex") {
    data$ECond[i] <- 5
  }
}

boxplot(data$SalePrice ~ data$ECond)

##Checking Numerical Corr  and Colinearity with new values: 

#Update Numeric attributes list
numeric_att <- data[which(sapply(data, is.numeric))]

#Checking for co-linearity and co-dependence
cor(numeric_att)
#corr         OverallQual  YearBuilt YearRemodAdd  TotalBsmtSF    X1stFlrSF  GrLivArea     FullBath
#OverallQual   1.00000000  0.5733340   0.55160539  0.537522541  0.475933049 0.59302078  0.551267391
#YearBuilt     0.57333396  1.0000000   0.59251163  0.393158750  0.283056369 0.19928607  0.467960368
#YearRemodAdd  0.55160539  0.5925116   1.00000000  0.292546085  0.241344125 0.28767718  0.438732682
#TotalBsmtSF   0.53752254  0.3931587   0.29254609  1.000000000  0.819393271 0.45500099  0.324885627
#X1stFlrSF     0.47593305  0.2830564   0.24134412  0.819393271  1.000000000 0.56608370  0.381437520
#GrLivArea     0.59302078  0.1992861   0.28767718  0.455000994  0.566083699 1.00000000  0.630283084
#FullBath      0.55126739  0.4679604   0.43873268  0.324885627  0.381437520 0.63028308  1.000000000
#TotRmsAbvGrd  0.42771967  0.0954013   0.19160299  0.286124832  0.409900501 0.82557645  0.554758945
#GarageCars    0.60099085  0.5378662   0.42057283  0.435279281  0.439725874 0.46732095  0.469619333
#GarageArea    0.56197990  0.4794597   0.37201234  0.486718984  0.489741434 0.46897069  0.405944028
#SalePrice     0.79106866  0.5232731   0.50743022  0.613904967  0.605967857 0.70861761  0.560880624
#NHood         0.67542960  0.6908120   0.51566814  0.461771612  0.416459681 0.41678820  0.509035327
#KQual         0.67400306  0.5299238   0.62513752  0.433835428  0.387781252 0.42076521  0.434423128
#ExQual        0.72617835  0.5990735   0.58815301  0.470310555  0.397552725 0.43595581  0.484505210
#MSZone        0.26956788  0.4618642   0.24098002  0.237889281  0.240828092 0.16064137  0.266646022
#BType         0.17675121  0.1329993   0.12827778  0.131417906  0.077387474 0.01662873 -0.044571286
#HStyle        0.29814522  0.3367999   0.29544509 -0.013251413 -0.071024392 0.38607996  0.356346456
#SCond         0.24981972  0.2633365   0.25601319  0.196606250  0.133689861 0.12120047  0.150333218
#Cair          0.27224769  0.3818050   0.29880903  0.208408642  0.147203725 0.09371136  0.109209510
#HQC           0.45710269  0.4491818   0.55034987  0.265739084  0.189852343 0.25462656  0.333654738
#ECond         0.01382636 -0.1037863   0.07495107 -0.006127493 -0.001564364 0.01653550 -0.003197331
#             TotRmsAbvGrd   GarageCars GarageArea  SalePrice       NHood      KQual     ExQual      MSZone
#OverallQual   0.427719672  0.600990851 0.56197990 0.79106866  0.67542960 0.67400306 0.72617835  0.26956788
#YearBuilt     0.095401296  0.537866193 0.47945968 0.52327306  0.69081203 0.52992376 0.59907354  0.46186425
#YearRemodAdd  0.191602993  0.420572827 0.37201234 0.50743022  0.51566814 0.62513752 0.58815301  0.24098002
#TotalBsmtSF   0.286124832  0.435279281 0.48671898 0.61390497  0.46177161 0.43383543 0.47031055  0.23788928
#X1stFlrSF     0.409900501  0.439725874 0.48974143 0.60596786  0.41645968 0.38778125 0.39755273  0.24082809
#GrLivArea     0.825576450  0.467320947 0.46897069 0.70861761  0.41678820 0.42076521 0.43595581  0.16064137
#FullBath      0.554758945  0.469619333 0.40594403 0.56088062  0.50903533 0.43442313 0.48450521  0.26664602
#TotRmsAbvGrd  1.000000000  0.362248162 0.33791527 0.53377887  0.28833479 0.28719083 0.29810062  0.12177606
#GarageCars    0.362248162  1.000000000 0.88261303 0.64047290  0.58512641 0.50976397 0.52664089  0.25337579
#GarageArea    0.337915274  0.882613027 1.00000000 0.62342290  0.53624006 0.48988835 0.49571683  0.25577284
#SalePrice     0.533778865  0.640472903 0.62342290 1.00000000  0.70761475 0.65981393 0.68267731  0.32466864
#NHood         0.288334785  0.585126406 0.53624006 0.70761475  1.00000000 0.56962833 0.64322708  0.57821554
#KQual         0.287190825  0.509763971 0.48988835 0.65981393  0.56962833 1.00000000 0.71671992  0.24688574
#ExQual        0.298100624  0.526640889 0.49571683 0.68267731  0.64322708 0.71671992 1.00000000  0.27482135
#MSZone        0.121776063  0.253375788 0.25577284 0.32466864  0.57821554 0.24688574 0.27482135  1.00000000
#BType        -0.072236870  0.129162210 0.13094978 0.17745005  0.17889689 0.18113539 0.17061624  0.15535257
#HStyle        0.349544505  0.273301155 0.22453311 0.29174383  0.31931553 0.22740629 0.25525836  0.19440875
#SCond         0.066086996  0.231401678 0.24688981 0.27761000  0.26360936 0.28555025 0.30153421  0.12077597
#Cair          0.034457555  0.233682854 0.23081727 0.25136718  0.27292354 0.25775115 0.20627556  0.25432118
#HQC           0.164941473  0.325392694 0.29544103 0.42763872  0.41819474 0.50440222 0.52004394  0.19982394
#ECond         0.001042347 -0.009326223 0.02028572 0.01887209 -0.06193527 0.05772373 0.00907309 -0.05430171
#               BType       HStyle       SCond       Cair        HQC        ECond
#OverallQual   0.176751211  0.298145220  0.24981972 0.27224769 0.45710269  0.013826355
#YearBuilt     0.132999346  0.336799925  0.26333650 0.38180504 0.44918181 -0.103786297
#YearRemodAdd  0.128277777  0.295445085  0.25601319 0.29880903 0.55034987  0.074951070
#TotalBsmtSF   0.131417906 -0.013251413  0.19660625 0.20840864 0.26573908 -0.006127493
#X1stFlrSF     0.077387474 -0.071024392  0.13368986 0.14720372 0.18985234 -0.001564364
#GrLivArea     0.016628731  0.386079963  0.12120047 0.09371136 0.25462656  0.016535501
#FullBath     -0.044571286  0.356346456  0.15033322 0.10920951 0.33365474 -0.003197331
#TotRmsAbvGrd -0.072236870  0.349544505  0.06608700 0.03445755 0.16494147  0.001042347
#GarageCars    0.129162210  0.273301155  0.23140168 0.23368285 0.32539269 -0.009326223
#GarageArea    0.130949779  0.224533107  0.24688981 0.23081727 0.29544103  0.020285720
#SalePrice     0.177450051  0.291743829  0.27761000 0.25136718 0.42763872  0.018872089
#NHood         0.178896895  0.319315535  0.26360936 0.27292354 0.41819474 -0.061935267
#KQual         0.181135386  0.227406287  0.28555025 0.25775115 0.50440222  0.057723726
#ExQual        0.170616236  0.255258364  0.30153421 0.20627556 0.52004394  0.009073090
#MSZone        0.155352569  0.194408752  0.12077597 0.25432118 0.19982394 -0.054301705
#BType         1.000000000 -0.002712872  0.11431249 0.23718421 0.16278078  0.089852222
#HStyle       -0.002712872  1.000000000  0.11346118 0.09184679 0.17681248 -0.044850912
#SCond         0.114312491  0.113461178  1.00000000 0.10620556 0.24630258 -0.034689215
#Cair          0.237184211  0.091846790  0.10620556 1.00000000 0.30633108  0.094516592
#HQC           0.162780782  0.176812482  0.24630258 0.30633108 1.00000000  0.055496673
#ECond         0.089852222 -0.044850912 -0.03468921 0.09451659 0.05549667  1.000000000

#In a more visual form:
corrplot.mixed(cor(numeric_att))

#The standard of co-linearity I set was for an R^2 of 0.6, based off of an accepted standard.  This correlates to
#a correlation of around 0.75 or greater.  The 3 pairs that met this standard are evaluated below:

#Since GarageCars had a slightly higher corellation to SalePrice, remove GarageArea
remove <- "GarageArea"
#TotBsmtSF has a slightly higher corellation, but I chose to eliminate it over X1stFlrSF (I believe the 1st Floor SF will better represent value)
remove <- append(remove, "TotalBsmtSF")
#Since GrLivArea has a much higher corellation to SalePrice, remove TotRmsAbvGrd
remove <- append(remove, "TotRmsAbvGrd")

data <- data[-which(colnames(data) %in% remove)]


##Outliers and Log Transformations##

#Removing outliers tuple 
#Looking at data, 9 tuples list full baths at 0.
#Having 0 Full Baths for these homes doesn't make sense.  Assuming the data is wrong, and removing tuples
data <- data[-c(which(data$FullBath == 0)), ] #Removes 9 tuples (tuple count @ 1450)
data$FullBath <- factor(data$FullBath, exclude = 0) #fixing the change in factor levels
data$FullBath <- as.numeric(data$FullBath)

#Doing Log Transformation on Sale Price
hist(data$SalePrice)
data$SalePrice.log <- log(data$SalePrice)
hist(data$SalePrice.log)



#Finding and removing an outlier of 1st Floor SF
plot(data$X1stFlrSF)
which(data$X1stFlrSF > 4000)
#[1] 1290

data <- data[-1290,] #Tuple count @1449
plot(data$X1stFlrSF)


#Log Transformation on X1stFlrSF
hist(data$X1stFlrSF)
data$X1stFlrSF.log <- log(data$X1stFlrSF)
hist(data$X1stFlrSF.log)


#Finding and removing outliers in GrLivArea
plot(data$GrLivArea ~ data$SalePrice) #3 Major outliers spotted
which(data$GrLivArea > 4000)
#[1]  521  687 1176

data <- data[-c(521, 687, 1176), ] #Tuple count @1446
plot(data$GrLivArea ~ data$SalePrice)

#Log Transformation of GrLivArea
hist(data$GrLivArea)
data$GrLivArea.log <- log(data$GrLivArea)
hist(data$GrLivArea.log)


##Sampling and CLT:##

#100 samples of 50:
num.samples <- 100
sample.size <- 50
saleprice.mean <- mean(data$SalePrice.log)
xbar <- numeric(num.samples)
ci.low <- numeric(num.samples)
ci.high <- numeric(num.samples)
sd.error <- sd(data$SalePrice.log)/sqrt(num.samples)
for (i in 1:num.samples){
       sample.data <- sample(data$SalePrice.log, sample.size)
       xbar[i] <- mean(sample.data)
       ci.low[i] <- xbar[i]-1.96*sd.error
       ci.high[i] <- xbar[i]+1.96*sd.error
   }

   
bind1 <- rbind(ci.low, ci.high)
bind2 <- rbind(1:num.samples, 1:num.samples)
matplot(bind1, bind2, type = "l", ylab = "Samples", xlab = "CI Ranges")
abline(v=saleprice.mean)
df <- data.frame("pop.mean" = saleprice.mean, "CI-Low"=ci.low, "CI-High"=ci.high)

#Which samples don't include the pop mean:
df[which(df$pop.mean < df$CI.Low | df$pop.mean > df$CI.High),]
#   pop.mean   CI.Low  CI.High
#1  12.02325 12.02847 12.18271
#10 12.02325 12.07649 12.23072
#44 12.02325 11.84551 11.99975
#48 12.02325 11.82607 11.98031
#56 12.02325 12.06165 12.21588
#71 12.02325 12.03205 12.18628
#72 12.02325 12.04810 12.20233
#77 12.02325 12.02934 12.18357
#89 12.02325 11.86450 12.01874
#98 12.02325 11.85817 12.01240

#200 samples of 100:
num.samples2 <- 200
sample.size2 <- 100
saleprice.mean <- mean(data$SalePrice.log)
xbar2 <- numeric(num.samples2)
ci.low2 <- numeric(num.samples2)
ci.high2 <- numeric(num.samples2)
sd.error2 <- sd(data$SalePrice.log)/sqrt(num.samples2)
for (i in 1:num.samples2){
  sample.data2 <- sample(data$SalePrice.log, sample.size2)
  xbar2[i] <- mean(sample.data2)
  ci.low2[i] <- xbar2[i]-1.96*sd.error2
  ci.high2[i] <- xbar2[i]+1.96*sd.error2
}


bind1 <- rbind(ci.low2, ci.high2)
bind2 <- rbind(1:num.samples2, 1:num.samples2)
matplot(bind1, bind2, type = "l", ylab = "Samples", xlab = "CI Ranges")
abline(v=saleprice.mean)
df2 <- data.frame("pop mean" = saleprice.mean, "CI-Low"=ci.low2, "CI-High"=ci.high2)

#Which samples don't include the pop mean:
df2[which(df2$pop.mean < df2$CI.Low | df2$pop.mean > df2$CI.High),]
#   pop.mean   CI.Low  CI.High
#1   12.02325 12.03554 12.14460
#2   12.02325 11.90473 12.01379
#11  12.02325 11.85097 11.96003
#12  12.02325 12.04627 12.15533
#19  12.02325 12.08522 12.19428
#22  12.02325 11.91345 12.02251
#25  12.02325 12.02813 12.13719
#30  12.02325 12.03108 12.14014
#34  12.02325 11.91335 12.02241
#35  12.02325 11.88533 11.99439
#46  12.02325 11.90016 12.00922
#58  12.02325 12.03489 12.14395
#77  12.02325 11.88850 11.99756
#84  12.02325 11.90675 12.01581
#97  12.02325 12.04360 12.15266
#102 12.02325 11.90794 12.01700
#106 12.02325 12.04721 12.15627
#110 12.02325 11.91217 12.02123
#115 12.02325 12.04643 12.15549
#116 12.02325 12.04537 12.15443
#118 12.02325 12.04772 12.15678
#133 12.02325 12.04352 12.15258
#135 12.02325 12.02685 12.13591
#136 12.02325 11.90152 12.01058
#145 12.02325 11.90145 12.01051
#148 12.02325 12.04207 12.15113
#152 12.02325 11.88555 11.99461
#166 12.02325 11.90830 12.01736
#167 12.02325 11.88583 11.99489
#183 12.02325 12.05315 12.16221
#192 12.02325 12.03835 12.14741


#20 samples of 20:
num.samples3 <- 20
sample.size3 <- 20
saleprice.mean <- mean(data$SalePrice.log)
xbar3 <- numeric(num.samples3)
ci.low3 <- numeric(num.samples3)
ci.high3 <- numeric(num.samples3)
sd.error3 <- sd(data$SalePrice.log)/sqrt(num.samples3)
for (i in 1:num.samples3){
  sample.data3 <- sample(data$SalePrice.log, sample.size3)
  xbar3[i] <- mean(sample.data3)
  ci.low3[i] <- xbar3[i]-1.96*sd.error3
  ci.high3[i] <- xbar3[i]+1.96*sd.error3
}


bind1 <- rbind(ci.low3, ci.high3)
bind2 <- rbind(1:num.samples3, 1:num.samples3)
matplot(bind1, bind2, type = "l", ylab = "Samples", xlab = "CI Ranges")
abline(v=saleprice.mean)
df3 <- data.frame("pop mean" = saleprice.mean, "CI-Low"=ci.low3, "CI-High"=ci.high3)

#Which samples don't include the pop mean:
df3[which(df3$pop.mean < df3$CI.Low | df3$pop.mean > df3$CI.High),]
#   pop.mean   CI.Low  CI.High
#9  12.02325 12.03006 12.37494
#18 12.02325 12.05027 12.39515

#####CLT in action####
hist(data$SalePrice.log)
mean(data$SalePrice.log)
#[1] 12.02325
sd(data$SalePrice.log)
#0.3934541

#1 Sample of 50:
sample.size4 <- 50
xbar4 <- numeric(sample.size4)
sample.data4 <- sample(data$SalePrice.log, sample.size4)
p50 <- hist(sample.data4)
sd(sample.data4)
#[1] 0.383461
mean(sample.data4)
#[1] 11.9595

#1 Sample of 200:
sample.size5 <- 200
xbar5 <- numeric(sample.size5)
sample.data5 <- sample(data$SalePrice.log, sample.size5)
p200 <- hist(sample.data5, 10)
sd(sample.data5)
#[1] 0.4021201
mean(sample.data5)
#[1] 12.03448

#1 Sample of 500:
sample.size6 <- 500
xbar6 <- numeric(sample.size6)
sample.data6 <- sample(data$SalePrice.log, sample.size6)
#sample.data6$fiveHundred <- "500"
p500 <- hist(sample.data6)
sd(sample.data6)
#[1] 0.372619
mean(sample.data6)
#[1] 12.00659

plot(p500, col=rgb(0,0,1,1/4),main = "Comparing Histogram of Differing Sample Sizes", xlab = "Log Sales Price")
plot(p200, col=rgb(1,0,0,1/4), add=T)
plot(p50, col =rgb(1,0,1,1/4), add=T)

###########End of Project Presentation Data#########################



##Below is some regression analysis done on this data, and will not be covered in the presentation during class.
##It was done to complete the Kaggle challenge from which this data set come from, so we thought we would 
#include it in the source code, since we did do the work.



##Exploring the data##

train_att <- c("GrLivArea.log", "X1stFlrSF.log", "GarageCars", "OverallQual", "FullBath", "YearBuilt", "Econd", "HQC", "CAir", "ExQual", "KQual", "SCond", "HStyle", "BType", "MSZone", "NHood", "SalePrice.log")
data.all<- subset(data, select = which(colnames(data) %in% train_att))
#Switch the order around so the class label is the last column:
data.all <- subset(data.all, select = c(1:12, 14,15, 13))


#Split training (60%) and test data (40%)
data.train <- data.all[1:floor(length(data.all[,1])*0.6),]
data.test <- data.all[(length(data.train[,1])+1):length(data.all[,1]),] 


#Function to test the RMSE value on test data
rmse_eval = function(actual, pred) {
  return (sqrt(sum((actual - pred)^2) / length(actual))) 
}


#Results of running Multiple Linear Regression on training data
reg1 <- lm(SalePrice.log~., data = data.train)
summary(reg1)

#Call:
#  lm(formula = SalePrice.log ~ ., data = data.train)

#Residuals:
#  Min       1Q   Median       3Q      Max 
#-0.81498 -0.07830  0.01236  0.08778  0.49604 

#Coefficients:
#                  Estimate Std. Error t value Pr(>|t|)    
#  (Intercept)    4.8820346  0.6028303   8.099 1.92e-15 ***
#  OverallQual    0.0734601  0.0067146  10.940  < 2e-16 ***
#  YearBuilt      0.0008601  0.0002782   3.092 0.002052 ** 
#  FullBath      -0.0230189  0.0138549  -1.661 0.096997 .  
#  GarageCars     0.0565687  0.0097749   5.787 1.01e-08 ***
#  NHood          0.0442945  0.0072682   6.094 1.66e-09 ***
#  MSZone         0.0337883  0.0079595   4.245 2.43e-05 ***
#  BType          0.0221757  0.0140872   1.574 0.115819    
#  HStyle        -0.0114143  0.0094874  -1.203 0.229272    
#  SCond          0.0394109  0.0116417   3.385 0.000743 ***
#  KQual          0.0584935  0.0117772   4.967 8.23e-07 ***
#  ExQual        -0.0063129  0.0149774  -0.421 0.673499    
#  HQC            0.0256137  0.0064746   3.956 8.25e-05 ***
#  X1stFlrSF.log  0.1664746  0.0224675   7.410 3.05e-13 ***
#  GrLivArea.log  0.4170317  0.0282454  14.765  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 0.1465 on 852 degrees of freedom
#Multiple R-squared:  0.8695,	Adjusted R-squared:  0.8673 
#F-statistic: 405.4 on 14 and 852 DF,  p-value: < 2.2e-16

#Training RMSE Calculations:
mse <- mean(residuals(reg1)^2)
rmse1 <- sqrt(mse)
rmse1
#0.1452628

#RMSE on test data:
reg1.predict <- predict(reg1, data.test)
reg1.rmse <- rmse_eval(reg1.predict, data.test$SalePrice.log)
reg1.rmse
#[1] 0.1473733


#LM Attempts, with the previous LM's as a guide as to what to include/exclude
reg2 <- lm(SalePrice.log~ OverallQual + YearBuilt + GarageCars + NHood + MSZone + SCond + KQual + HQC + X1stFlrSF.log + GrLivArea.log, data = data.train)
summary(reg2)

#Call:
#  lm(formula = SalePrice.log ~ OverallQual + YearBuilt + GarageCars + 
#       NHood + MSZone + SCond + KQual + HQC + X1stFlrSF.log + GrLivArea.log, 
#     data = data.train)

#Residuals:
#  Min       1Q   Median       3Q      Max 
#-0.81140 -0.07604  0.01235  0.08952  0.46940 

#Coefficients:
#                 Estimate Std. Error t value Pr(>|t|)    
#  (Intercept)   5.4903263  0.5578664   9.842  < 2e-16 ***
#  OverallQual   0.0740926  0.0064666  11.458  < 2e-16 ***
#  YearBuilt     0.0006267  0.0002623   2.389 0.017100 *  
#  GarageCars    0.0572779  0.0097442   5.878 5.94e-09 ***
#  NHood         0.0420817  0.0070867   5.938 4.19e-09 ***
#  MSZone        0.0356403  0.0078855   4.520 7.06e-06 ***
#  SCond         0.0401235  0.0116385   3.447 0.000593 ***
#  KQual         0.0581652  0.0109876   5.294 1.52e-07 ***
#  HQC           0.0256156  0.0064037   4.000 6.88e-05 ***
#  X1stFlrSF.log 0.1843498  0.0195527   9.428  < 2e-16 ***
#  GrLivArea.log 0.3752617  0.0217759  17.233  < 2e-16 ***
#  ---
#  Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1

#Residual standard error: 0.1469 on 856 degrees of freedom
#Multiple R-squared:  0.8683,	Adjusted R-squared:  0.8667 
#F-statistic: 564.2 on 10 and 856 DF,  p-value: < 2.2e-16

#Training RMSE
mse <- mean(residuals(reg2)^2)
rmse2 <- sqrt(mse)
rmse2
#[1] 0.1459395

#RMSE on actual test data:
reg2.predict <- predict(reg2, data.test)
reg2.rmse <- rmse_eval(reg2.predict, data.test$SalePrice.log)
reg2.rmse
#[1] 0.1492712


#XGBoost Model Training:
train_x <- xgb.DMatrix(as.matrix(data.train), label = data.train$SalePrice.log)
test_x <- xgb.DMatrix(as.matrix(data.test))

xgb_params <- list(
  booster = 'gbtree',
  objective = 'reg:linear',
  colsample_bytree=0.512,
  eta=0.123,
  max_depth=8,
  min_child_weight=1.25,
  alpha=0.9,
  lambda=0.6,
  subsample=0.734,
  seed=5,
  silent=TRUE)

#Using k-folds technique to show how the training model will predict on potential test data (cross validation)
train_xgb = xgb.cv(xgb_params, train_x, nrounds = 2000, nfold = 4)
train_xgb

##### xgb.cv 4-folds

#iter train_rmse_mean train_rmse_std test_rmse_mean test_rmse_std
#1      10.1232660    0.006196933     10.1232442    0.01847364
#2       8.8814865    0.004946893      8.8814628    0.01957221
#3       7.7930140    0.005427486      7.7929877    0.01912428
#4       6.8387332    0.005073198      6.8387045    0.01948291
#5       6.0016728    0.004306680      6.0016407    0.02042116
#---                                                                 
#1996       0.1103818    0.001418320      0.2036537    0.01288824
#1997       0.1103750    0.001417858      0.2036495    0.01286756
#1998       0.1103690    0.001422625      0.2036345    0.01283468
#1999       0.1103660    0.001424034      0.2036190    0.01283953
#2000       0.1103640    0.001423274      0.2036152    0.01287118

#Training the model:
set.seed(123)
xgb.model_train = xgb.train(xgb_params, train_x, nrounds = 600)

#Developing the predicted numbers:
xgb.predict <- predict(xgb.model_train, test_x)

#RMSE on test data:
xgb.rmse <- rmse_eval(xgb.predict, data.test$SalePrice.log)
xgb.rmse
#[1] 0.175888


#Random Forest Evaluation:
set.seed(123)
rand_forest <- randomForest(SalePrice.log~., data = data.train)
rand_forest

#Call:
#  randomForest(formula = SalePrice.log ~ ., data = data.train) 
#         Type of random forest: regression
#               Number of trees: 500
#No. of variables tried at each split: 4

#         Mean of squared residuals: 0.02301841
#               % Var explained: 85.76

#Training RMSE
sqrt(.02309805)
#[1] 0.1519804

#RMSE on test data
rand_forest.predict <- predict(rand_forest, data.test)
rand.forest_rmse <- rmse_eval(rand_forest.predict, data.test$SalePrice.log)
rand.forest_rmse
#[1] 0.1393626


# ridge, lasso, elasticnet (done together because they are only different variants of  
# regression analysis paths on the same algorithm type: Generalized Linear Models)

ridge.glm.cv = cv.glmnet(as.matrix(data.train), data.train$SalePrice.log, alpha = 0)
lasso.glm.cv = cv.glmnet(as.matrix(data.train), data.train$SalePrice.log, alpha = 1)
net.glm.cv = cv.glmnet(as.matrix(data.train), data.train$SalePrice.log, alpha = 0.001)

# use the lamdba that minimizes the error
ridge.penalty = ridge.glm.cv$lambda.min
lasso.penalty = lasso.glm.cv$lambda.min
net.penalty = net.glm.cv$lambda.min

#Train models
ridge.glm <- glmnet(x = as.matrix(data.train), y = data.train$SalePrice.log, alpha = 0, lambda = ridge.penalty)
lasso.glm = glmnet(x = as.matrix(data.train), y = data.train$SalePrice.log, alpha = 1, lambda = lasso.penalty)
net.glm = glmnet(x = as.matrix(data.train), y = data.train$SalePrice.log, alpha = 0.001, lambda = net.penalty)


#RMSE on test data
ridge.pred = predict(ridge.glm, as.matrix(data.test))
lasso.pred = predict(lasso.glm, as.matrix(data.test))
net.pred = predict(net.glm, as.matrix(data.test))

ridge.rmse <- rmse_eval(data.test$SalePrice.log, ridge.pred)
lasso.rmse <- rmse_eval(data.test$SalePrice.log, lasso.pred)
net.rmse <- rmse_eval(data.test$SalePrice.log, net.pred)

ridge.rmse 
#[1] 0.06113154
lasso.rmse 
#[1] 0.0121586
net.rmse
#[1] 0.06108604

