#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")
n.folds <- 10
alg.name <- "glmnet"

fn.opt.pred <- function(pars, data) {
  pars.m <- matrix(rep(pars,each=nrow(data)),nrow=nrow(data))
  rowSums(data*pars.m) 
} 
fn.opt <- function(pars) {     
  -auc(pred.train$action, fn.opt.pred(pars, pred.train[,cols])) 
} 

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

tt <- rbind(train,test)
flist <- c("resource","mgr","role1","role2","dept","title","desc","fam")
cols2 <- combn(flist,2)
for (i in 1:ncol(cols2)) {
  col <- cols2[,i]
  tt.feature <- tt[,list(feature=.GRP),by=col]
  tt <- merge(tt,tt.feature,by=col,all.x=TRUE)
  setnames(tt,"feature",paste(col,collapse="_"))
}

cols3 <- combn(flist,3)
for (i in 1:ncol(cols3)) {
  col <- cols3[,i]
  tt.feature <- tt[,list(feature=.GRP),by=col]
  tt <- merge(tt,tt.feature,by=col,all.x=TRUE)
  setnames(tt,"feature",paste(col,collapse="_"))
}

flist <- setdiff(colnames(tt),c("id","action"))
for (col in flist) {
  feature <- tt[[col]]
  freq <- table(feature)
  levels_lowfreq <- as.numeric(names(freq[which(freq==1)]))
  ix <- which(feature %in% levels_lowfreq)
  feature[ix] <- min(levels_lowfreq)
  tt[[col]] <- as.factor(feature)
}

train.lb <- data.frame(tt[action>=0])
test.lb <- data.frame(tt[action<0])
train.lb[is.na(train.lb)] <- -1
test.lb[is.na(test.lb)] <- -1

set.seed(3847569)
data.cv.folds <- cvFolds(nrow(train.lb), K = n.folds, type="interleaved")
cat("Instance CV distribution: \n")
print(table(data.cv.folds$which))

for (iter in 1:1) {
  chosen <- sample(flist,4)
  tt.all <- sparse.model.matrix(~ . -1, data = rbind(train.lb[,chosen],test.lb[,chosen]))
  train.lb.sparse <- tt.all[1:nrow(train.lb),]
  test.lb.sparse <- tt.all[(nrow(train.lb)+1):nrow(tt.all),]
  
  fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
  glmnet.pred <- foreach(k=1:(data.cv.folds$K+1),.errorhandling="remove") %dopar% {
    
    file.name <- "output_glm"
    fn.init.worker(paste(file.name,k,sep=""))
    library(glmnet)
    
    if (k <= data.cv.folds$K) {
      data.glm <- list()
      data.glm$data.tr <- train.lb.sparse[which(data.cv.folds$which!=k),]
      data.glm$target.tr <- train.lb[which(data.cv.folds$which!=k),"action"]
      data.glm$data.test <- train.lb.sparse[which(data.cv.folds$which==k),]
    } else {
      data.glm <- list()
      data.glm$data.tr <- train.lb.sparse
      data.glm$target.tr <- train.lb[,"action"]
      data.glm$data.test <- test.lb.sparse
    }

    model.glmnet <- cv.glmnet(
      x = data.glm$data.tr,
      y = data.glm$target.tr,
      family = "binomial",
      standardize = F,
      alpha = 0.5,
      type.measure = "auc",
      intercept = T,
      nfolds = 10)
    
    pred <- predict(model.glmnet, data.glm$data.test, type="response", s = "lambda.min")[,1]
    
    fn.clean.worker()
    
    pred
  }
  fn.kill.wk()

  #extract prediction
  #on train
  pred.glm.train <- rep(0,nrow(train.lb))
  for (k in 1:data.cv.folds$K) {
    pred.glm.train[which(data.cv.folds$which==k)] <- glmnet.pred[[k]]
  }
  print (auc(train.lb[,'action'],pred.glm.train))
  #on test
  pred.glm.test <- glmnet.pred[[data.cv.folds$K+1]]
}

pred.glm.train0 <- pred.glm.train
pred.glm.test0 <- pred.glm.test
for (iter in 2:300) {
  print (paste0("iter = ",iter))
  chosen <- sample(flist,4)
  tt.all <- sparse.model.matrix(~ . -1, data = rbind(train.lb[,chosen],test.lb[,chosen]))
  train.lb.sparse <- tt.all[1:nrow(train.lb),]
  test.lb.sparse <- tt.all[(nrow(train.lb)+1):nrow(tt.all),]
  
  fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
  glmnet.pred <- foreach(k=1:(data.cv.folds$K+1),.errorhandling="remove") %dopar% {
    
    file.name <- "output_glm"
    fn.init.worker(paste(file.name,k,sep=""))
    library(glmnet)
    
    if (k <= data.cv.folds$K) {
      data.glm <- list()
      data.glm$data.tr <- train.lb.sparse[which(data.cv.folds$which!=k),]
      data.glm$target.tr <- train.lb[which(data.cv.folds$which!=k),"action"]
      data.glm$data.test <- train.lb.sparse[which(data.cv.folds$which==k),]
    } else {
      data.glm <- list()
      data.glm$data.tr <- train.lb.sparse
      data.glm$target.tr <- train.lb[,"action"]
      data.glm$data.test <- test.lb.sparse
    }
    
    model.glmnet <- cv.glmnet(
      x = data.glm$data.tr,
      y = data.glm$target.tr,
      family = "binomial",
      standardize = F,
      alpha = 0.5,
      type.measure = "auc",
      intercept = T,
      nfolds = 10)
    
    pred <- predict(model.glmnet, data.glm$data.test, type="response", s = "lambda.min")[,1]
    
    fn.clean.worker()
    
    pred
  }
  fn.kill.wk()
  
  #extract prediction
  #on train
  pred.glm.train <- rep(0,nrow(train.lb))
  for (k in 1:data.cv.folds$K) {
    pred.glm.train[which(data.cv.folds$which==k)] <- glmnet.pred[[k]]
  }
  print (paste0("current auc: ",auc(train.lb[,'action'],pred.glm.train)))
  #on test
  pred.glm.test <- glmnet.pred[[data.cv.folds$K+1]]
  
  #optimization
  pred.train <- data.frame(action=train.lb$action, prev=pred.glm.train0,cur=pred.glm.train)
  pred.test <- data.frame(prev=pred.glm.test0,cur=pred.glm.test)
  cols <- c("prev", "cur") 
  pars <- rep(1/length(cols),length(cols)) 
  opt.result <- optim(pars, fn.opt, control = list(trace = F))
  pred.glm.train0 <- fn.opt.pred(opt.result$par, pred.train[,cols])
  pred.glm.test0 <- fn.opt.pred(opt.result$par, pred.test[,cols])
  print (paste0("auc after optim: ",auc(pred.train$action, pred.glm.train0)))
}
pred.glm.train0 <- data.frame(id = train.lb$id, pred = pred.glm.train0)
pred.glm.train0 <- pred.glm.train0[order(pred.glm.train0$id),]
pred.train <- pred.glm.train0[,"pred",drop=FALSE]

pred.glm.test0 <- data.frame(id = test.lb$id, pred = pred.glm.test0)
pred.glm.test0 <- pred.glm.test0[order(pred.glm.test0$id),]
pred.test <- pred.glm.test0[,"pred",drop=FALSE]

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))