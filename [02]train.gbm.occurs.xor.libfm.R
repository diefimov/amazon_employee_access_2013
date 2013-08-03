#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")
n.folds <- 10
alg.name <- "gbm_occurs_xor_libfm"

tic()
cat("Loading csv data... ")
data.tr <- read.csv(fn.in.file("train.csv"))
data.test <- read.csv(fn.in.file("test.csv"))
data.test$id <- NULL
data.test$ACTION <- NA
data.test <- data.test[, colnames(data.tr)]
data.all <- rbind(data.tr, data.test)
for (col.name in colnames(data.all)[-1]) {
  data.all[[col.name]] <- factor(data.all[[col.name]])
}
data.all$ROLE_TITLE <- NULL
data.tr <- data.all[!is.na(data.all$ACTION),]
data.test <- data.all[is.na(data.all$ACTION),]
rownames(data.test) <- 1:nrow(data.test)
toc()

tic()
cat("Building cv... ")
data.cv.folds <- fn.cv.folds(nrow(data.tr), K = n.folds, seed = 3764743)
cat("done \n")
toc()

fn.load.data("data.tr.occurs")
fn.load.data("data.test.occurs")

data.tr <- cbind(data.tr[,"ACTION", drop = F], data.tr.occurs)
data.test <- cbind(data.test[,"ACTION", drop = F], data.test.occurs)

load("output-R/libfm.RData")
if (ncol(pred.train)>1) {
	print ("Something wrong with libfm model")
} else {
	colnames(pred.train) <- "libfm"
	colnames(pred.test) <- "libfm"
}
data.tr <- cbind(data.tr,pred.train[,"libfm",drop=F])
data.test <- cbind(data.test,pred.test[,"libfm",drop=F])

action0 <- data.tr$ACTION
new_action <- rep(0,nrow(data.tr))
new_action[which(data.tr$libfm>0.9 & action0==1)] <- 1
new_action[which(data.tr$libfm<=0.9 & action0==0)] <- 1
data.tr$ACTION <- new_action

#############################################################
# train using gbm.occurs with libfm
#############################################################
fn.register.wk()
gbm.occurs.pred <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  
  data.gbm.occurs <- list()
  
  val.select <-  fn.cv.which(data.cv.folds, k)
  
  data.gbm.occurs$log <- paste0("gbm_occurs_",k)
  data.gbm.occurs$log.full <- paste0("log/",data.gbm.occurs$log, ".log")
  
  #fn.init.worker(data.gbm.occurs$log)
  
  data.gbm.occurs$tr.idx <- which(!val.select)
  data.gbm.occurs$tr.x <- data.tr[data.gbm.occurs$tr.idx,-1]
  data.gbm.occurs$tr.y <- data.tr$ACTION[data.gbm.occurs$tr.idx]
  data.gbm.occurs$val.idx <- which(val.select)
  data.gbm.occurs$val.x <- data.tr[data.gbm.occurs$val.idx,-1]
  data.gbm.occurs$val.y <- data.tr$ACTION[data.gbm.occurs$val.idx]
  data.gbm.occurs$test.x <- data.test[,-1]
  
  library("gbm")
  library("Metrics")
  
   model.gbm = gbm.fit(
       x = data.gbm.occurs$tr.x,
       y = data.gbm.occurs$tr.y,
       distribution = "bernoulli",
       n.trees = 3000,
       shrinkage = 0.05,
       interaction.depth = 20) 
    
  #print(model.gbm)
  #print(summary(model.gbm, plotit=F))
  
  pred.trees <- model.gbm$n.trees
  
  data.pred <- NULL
  if (NROW(data.gbm.occurs$val.x) > 0) {
    data.pred <- data.frame(
      datatype = "tr",
      test.idx = data.gbm.occurs$val.idx,
      pred = predict(model.gbm, data.gbm.occurs$val.x, pred.trees, type="response")
      )
    #print(auc(data.gbm.occurs$val.y,  data.pred$pred))
  }
  
  data.pred.test <- data.frame(
    datatype = "test",
    test.idx = 1:nrow(data.test),
    pred = predict(model.gbm, data.gbm.occurs$test.x, pred.trees, type="response")
    )
  rm(model.gbm)
  #fn.clean.worker()
  
  rbind(data.pred, data.pred.test)
}
fn.kill.wk()

#############################################################
# extract prediction
#############################################################

data.tr.gbm.occurs <- fn.extract.tr(gbm.occurs.pred)
resdf <- data.frame(action=action0,similarity=data.tr.gbm.occurs[,1],libfm=data.tr$libfm)
resdf[,"pred"] <- resdf$libfm
ix <- which(resdf$libfm<=0.9)
resdf[ix,"pred"] <- 1-resdf$similarity[ix]
ix <- which(resdf$libfm>0.9)
resdf[ix,"pred"] <- resdf$similarity[ix]
auc(resdf$action,resdf$pred)
pred.train <- resdf[,"pred",drop=FALSE]
auc(action0,pred.train$pred)

data.test.gbm.occurs <- fn.extract.test(gbm.occurs.pred)
resdf <- data.frame(similarity=data.test.gbm.occurs[,1],libfm=data.test$libfm)
resdf[,"pred"] <- resdf$libfm
ix <- which(resdf$libfm<=0.9)
resdf[ix,"pred"] <- 1-resdf$similarity[ix]
ix <- which(resdf$libfm>0.9)
resdf[ix,"pred"] <- resdf$similarity[ix]
pred.test <- resdf[,"pred",drop=FALSE]

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))