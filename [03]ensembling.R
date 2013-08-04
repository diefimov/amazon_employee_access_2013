#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")
algs <- c("gbm_freq_hom2","gbm_freq_hom3","gbm_lme","gbm_occurs","glmnet","glmnet2","lr","lr2","rf_freq","gbm_freq_hom5","gbm_occurs_xor_libfm") 

colClasses <- c("numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric","numeric")
train <- read.delim("data/train.csv",header=TRUE,sep=",",colClasses=colClasses,stringsAsFactors = FALSE)
train <- train[,1:(ncol(train)-1)]
n <- ncol(train)
train[,'id'] <- seq(500000,500000+nrow(train)-1,1)
train <- cbind(train[,'id',drop=FALSE],train[,1:n])
train <- data.table(train)
setnames(train,c("id","action","resource","mgr","role1","role2","dept","title","desc","fam"))

test <- read.delim("data/test.csv",header=TRUE,sep=",",colClasses=colClasses,stringsAsFactors = FALSE)
test <- test[,1:(ncol(test)-1)]
n <- ncol(test)
test[,'action'] <- -1
test <- cbind(test[,'id',drop=FALSE],test[,'action',drop=FALSE],test[,2:n])
test <- data.table(test)
setnames(test,c("id","action","resource","mgr","role1","role2","dept","title","desc","fam"))

pred.train.ens <- data.frame(action = train[,action])
pred.test.ens <- data.frame(id = test[,id])
for (alg in algs) {
	load(paste0("output-R/",alg,".RData"))
	colnames(pred.train) <- alg
	colnames(pred.test) <- alg
	pred.train.ens <- cbind(pred.train.ens,pred.train)
	pred.test.ens <- cbind(pred.test.ens,pred.test)
}

#find best combination using expand.grid
a <- as.matrix(expand.grid( rep(list(0:3),length(algs)) ))
a <- a[which(rowSums(a)>0),]
coeffs <- findBestCombination(pred.train.ens[,algs],pred.train.ens[,"action"],a)
print (coeffs)
ens.train <- as.matrix(pred.train.ens[,algs]) %*% (coeffs/sum(coeffs))
print (auc(pred.train.ens$action,ens.train))
ens.test <- as.matrix(pred.test.ens[,algs]) %*% (coeffs/sum(coeffs))

subm <- data.frame(Id=pred.test.ens$id,Action=ens.test)
subm <- subm[order(subm$Id),]
write.csv(subm,file="prediction_expandgrid.csv",quote=FALSE,row.names=FALSE)

#find best combination using optim
fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m) 
} 
fn.opt <- function(pars) {     
  -auc(pred.train.ens$action, fn.opt.pred(pars, pred.train.ens[,algs])) 
} 
pars <- rep(1/length(algs),length(algs)) 
opt.result <- optim(pars, fn.opt, control = list(trace = T))
ens.train <- fn.opt.pred(opt.result$par, pred.train.ens[,algs])
ens.test <- fn.opt.pred(opt.result$par, pred.test.ens[,algs])
auc(pred.train.ens[,"action"],ens.train)

subm <- data.frame(Id=pred.test.ens$id,Action=ens.test)
subm <- subm[order(subm$Id),]
write.csv(subm,file="prediction_optim.csv",quote=FALSE,row.names=FALSE)

#the best linear combination
algs <- c("gbm_freq_hom2","gbm_freq_hom3","gbm_lme","gbm_occurs","glmnet","glmnet2","lr","lr2","rf_freq","gbm_freq_hom5","gbm_occurs_xor_libfm")
coeffs <- c(1, 2, 4, 4, 0.5, 0.5, 1.5, 1.5, 1, 1, 2)
ens.train <- as.matrix(pred.train.ens[,algs]) %*% (coeffs/sum(coeffs))
auc(pred.train.ens[,"action"],ens.train)
ens.test <- as.matrix(pred.test.ens[,algs]) %*% (coeffs/sum(coeffs))

subm <- data.frame(Id=pred.test.ens$id,Action=ens.test)
subm <- subm[order(subm$Id),]
write.csv(subm,file="prediction_manual.csv",quote=FALSE,row.names=FALSE)
