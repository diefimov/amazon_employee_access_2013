#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")

#find best combination using expand.grid
cols <- c("rf_freq","gbm_occurs","gbm_lme","lr2_100_2","gbm_freq_hom3","gbm_freq_hom5","gbm_occurs_xor_libfm","glmnet2") 
a <- as.matrix(expand.grid( rep(list(0:7),length(cols)) ))
a <- a[which(rowSums(a)>0),]
coeffs <- findBestCombination(pred.train[,cols],pred.train[,"action"],a)
print (coeffs)
ens.train <- as.matrix(pred.train[,cols]) %*% (coeffs/sum(coeffs))
print (auc(pred.train$action,ens.train))
ens.test <- as.matrix(pred.test[,cols]) %*% (coeffs/sum(coeffs))

#find best combination using optim
cols <- c("rf_freq","gbm_freq_hom3","gbm_freq_hom5","gbm_occurs","gbm_lme","lr2_100_2","glmnet")
fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m) 
} 
fn.opt <- function(pars) {     
  -auc(pred.train$action, fn.opt.pred(pars, pred.train[,cols])) 
} 
pars <- rep(1/length(cols),length(cols)) 
opt.result <- optim(pars, fn.opt, control = list(trace = T))
ens.train <- fn.opt.pred(opt.result$par, pred.train[,cols])
ens.test <- fn.opt.pred(opt.result$par, pred.test[,cols])
auc(pred.train[,"action"],ens.train)

#the best linear combination
load("D:/Dropbox/Eclipse/Amazon2013/pred.dmitry.RData")
load("D:/Dropbox/Eclipse/Amazon2013/pred.lucas.RData")
pred.train <- cbind(pred.train.dmitry,pred.train.lucas[,-1])
pred.test <- cbind(pred.test.dmitry,pred.test.lucas)
pred.test[,"id"] <- test[,id]

pred <- (2*pred.test$gbm_freq_hom3_100 + 1*pred.test$gbm_freq_hom2_100 + 1*pred.test$rf_freq + 1*pred.test$gbm_freq_hom5_100 + 4*pred.test$gbm_occurs + 4*pred.test$gbm_lme + 3*(pred.test$lr2_100_2+pred.test$lr2_100)/2 + 1*(pred.test$glmnet+pred.test$glmnet2)/2 + 2*pred.test$gbm_occurs_xor_libfm)/19 #0.92679

subm <- data.frame(Id=pred.test$id,Action=pred)
subm <- subm[order(subm$Id),]
write.csv(subm,file="prediction.csv",quote=FALSE,row.names=FALSE)
