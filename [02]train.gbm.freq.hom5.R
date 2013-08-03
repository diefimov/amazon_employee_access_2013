#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")
n.folds <- 10
alg.name <- "gbm_freq_hom5"

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

###################################################################
###### FEATURES ENGINEERING #######################################
###################################################################

tt <- rbind(train,test)
tt.person <- tt[,list(person=.GRP),by=c("mgr","role1","role2","dept","title","desc","fam")]
tt <- merge(tt,tt.person,by=c("mgr","role1","role2","dept","title","desc","fam"),all.x=TRUE)

#uperson - united person
tt.uperson <- tt[,list(uperson=.GRP),by=c("role1","role2","dept","title","desc","fam")]
tt <- merge(tt,tt.uperson,by=c("role1","role2","dept","title","desc","fam"),all.x=TRUE)

#role3
tt.role3 <- tt[,list(role3=.GRP),by=c("role1","role2")]
tt <- merge(tt,tt.role3,by=c("role1","role2"),all.x=TRUE)

tt$role1 <- NULL
tt$fam <- NULL

#nresource_person - number of requested resources for the person
tt.quant <- tt[,list(nresource_person = length(unique(resource))),by=c("person")]
tt <- merge(tt,tt.quant,by=c("person"),all.x=TRUE)

#nperson_resource - number of persons requested for the resource
tt.quant <- tt[,list(nperson_resource = length(unique(person))),by=c("resource")]
tt <- merge(tt,tt.quant,by=c("resource"),all.x=TRUE)

#nperson_resource_dept - number of requests for the resource in department
tt.quant <- tt[,list(nperson_resource_dept = length(unique(person))),by=c("resource","dept")]
tt <- merge(tt,tt.quant,by=c("resource","dept"),all.x=TRUE)

#nmgr_resource - number of managers requested for the resource
tt.quant <- tt[,list(nmgr_resource = length(unique(mgr))),by=c("resource")]
tt <- merge(tt,tt.quant,by=c("resource"),all.x=TRUE)

#nuperson_resource - number of upersons for the resource
tt.quant <- tt[,list(nuperson_resource = length(unique(uperson))),by=c("resource")]
tt <- merge(tt,tt.quant,by=c("resource"),all.x=TRUE)

#nuperson_resource_dept - number of upersons for the resource in department
tt.quant <- tt[,list(nuperson_resource_dept = length(unique(uperson))),by=c("resource","dept")]
tt <- merge(tt,tt.quant,by=c("resource","dept"),all.x=TRUE)

load("output-R/libfm.RData")
if (ncol(pred.train)>1) {
	print ("Something wrong with libfm model")
} else {
	colnames(pred.train) <- "libfm"
	colnames(pred.test) <- "libfm"
}
pred.train[,"id"] <- train[,id]
pred.test[,"id"] <- test[,id]
pred <- rbind(pred.train[,c("id","libfm")], pred.test[,c("id","libfm")])
tt <- merge(tt,pred,by="id",all.x=TRUE)

###################################################################
###### CV AND PREDICTION ##########################################
###################################################################

train.lb <- data.frame(tt[action>=0])
test.lb <- data.frame(tt[action<0])
train.lb[is.na(train.lb)] <- -1
test.lb[is.na(test.lb)] <- -1

action0 <- train.lb$action
new_action <- rep(0,nrow(train.lb))
new_action[which(train.lb$libfm>0.9 & action0==1)] <- 1
new_action[which(train.lb$libfm<=0.9 & action0==0)] <- 1
train.lb$action <- new_action

set.seed(3847569)
data.cv.folds <- cvFolds(nrow(train.lb), K = n.folds, type="interleaved")
cat("Instance CV distribution: \n")
print(table(data.cv.folds$which))

###################################################################
###### GBM ########################################################
###################################################################

flist <- setdiff(colnames(train.lb),c("id"))
fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
gbm.pred <- foreach(k=1:(data.cv.folds$K+1),.errorhandling="remove") %dopar% {
  file.name <- "output_gbm5"
  fn.init.worker(paste(file.name,k,sep=""))
  
  if (k <= data.cv.folds$K) {
    data.gbm <- list()
    data.gbm$tr.idx <- which(data.cv.folds$which!=k)
    data.gbm$test.idx <-  which(data.cv.folds$which==k)
    data.gbm$data.tr <- train.lb[data.gbm$tr.idx,]
    data.gbm$data.test <- train.lb[data.gbm$test.idx,]
    data.gbm$data.type <- "tr" 
    n.trees0 <- 100
  } else {
    data.gbm <- list()
    data.gbm$tr.idx <- c(1:nrow(train.lb))
    data.gbm$test.idx <-  c(1:nrow(test.lb))
    data.gbm$data.tr <- train.lb
    data.gbm$data.test <- test.lb
    data.gbm$data.type <- "te" 
    n.trees0 <- 4000
  }
  
  library(gbm)
  set.seed(1532)
  
  model.gbm <- gbm(
    action~.,
    data = data.gbm$data.tr[,flist],
    distribution = "bernoulli",
    n.trees = n.trees0,
    shrinkage = 0.03,
    interaction.depth = 5,
    train.fraction = 1.0,
    bag.fraction = 0.5,
    n.minobsinnode = 10,
    keep.data = T,
    cv.folds = 0,
    verbose = F)
  
  if (k <= data.cv.folds$K) {
    library(Metrics)
    for (pred.trees in seq(100,4000,100)) {
      pred <- predict(model.gbm,
                      data.gbm$data.test[,flist], 
                      n.trees = pred.trees,
                      type = "response")
      print (paste("ntree =",pred.trees,": error =",auc(data.gbm$data.test[,'action'],pred)))
      model.gbm <- gbm.more(model.gbm,n.new.trees=100)
    }
    print(summary(model.gbm, plotit = F))
    pred.test.single <- predict(model.gbm,
                                test.lb[,flist], 
                                n.trees = 4000,
                                type = "response")
  } else {
    pred <- predict(model.gbm,
                    data.gbm$data.test[,flist], 
                    n.trees = n.trees0,
                    type = "response")
    pred.test.single <- pred
  }
  fn.clean.worker()
  list(pred, pred.test.single)
}
fn.kill.wk()

#extract prediction
#on train (not homogeneous)
pred.gbm.train <- rep(0,nrow(train.lb))
for (k in 1:data.cv.folds$K) {
  pred.gbm.train[which(data.cv.folds$which==k)] <- gbm.pred[[k]][[1]]
}
print (auc(train.lb[,'action'],pred.gbm.train))

resdf <- data.frame(id=train.lb$id,action=action0,similarity=pred.gbm.train,libfm=train.lb$libfm)
resdf[,"pred"] <- resdf$libfm
ix <- which(resdf$libfm<=0.9)
resdf[ix,"pred"] <- 1-resdf$similarity[ix]
ix <- which(resdf$libfm>0.9)
resdf[ix,"pred"] <- resdf$similarity[ix]
auc(resdf$action,resdf$pred)
resdf <- resdf[order(resdf$id),]
pred.gbm.train <- resdf[,c("id","pred")]
auc(train[,action],pred.gbm.train$pred)

#on test (homogeneous)
pred.gbm.test.hom <- rep(0,nrow(test.lb))
for (k in 1:data.cv.folds$K) {
  pred.gbm.test.hom <- pred.gbm.test.hom + gbm.pred[[k]][[2]]
}
resdf <- data.frame(id=test.lb$id,similarity=pred.gbm.test.hom/data.cv.folds$K,libfm=test.lb$libfm)
resdf[,"pred"] <- resdf$libfm
ix <- which(resdf$libfm<=0.9)
resdf[ix,"pred"] <- 1-resdf$similarity[ix]
ix <- which(resdf$libfm>0.9)
resdf[ix,"pred"] <- resdf$similarity[ix]
resdf <- resdf[order(resdf$id),]
pred.gbm.test.hom <- resdf[,c("id","pred")]
pred.gbm.test.hom <- pred.gbm.test.hom[order(pred.gbm.test.hom$id),]
pred.test <- pred.gbm.test.hom[,"pred",drop=FALSE]

#homogeneous gbm on train
pred.gbm.train.hom <- rep(0,nrow(train.lb))
for (iter in 1:data.cv.folds$K) {
  print (paste("iter:",iter))
  
  train.cv <- train.lb[which(data.cv.folds$which!=iter),]
  test.cv <- train.lb[which(data.cv.folds$which==iter),]
  data.cv.folds.cv <- cvFolds(nrow(train.cv), K = n.folds)
  cat("Instance CV distribution: \n")
  print(table(data.cv.folds.cv$which))
  
  flist <- setdiff(colnames(train.cv),c("id"))
  fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
  gbm.pred <- foreach(k=1:(data.cv.folds.cv$K),.errorhandling="remove") %dopar% {
    file.name <- "output_gbm5"
    fn.init.worker(paste(file.name,k,sep=""))
    
    if (k <= data.cv.folds.cv$K) {
      data.gbm <- list()
      data.gbm$tr.idx <- which(data.cv.folds.cv$which!=k)
      data.gbm$test.idx <-  which(data.cv.folds.cv$which==k)
      data.gbm$data.tr <- train.cv[data.gbm$tr.idx,]
      data.gbm$data.test <- train.cv[data.gbm$test.idx,]
      data.gbm$data.type <- "tr" 
      n.trees0 <- 100
    } else {
      data.gbm <- list()
      data.gbm$tr.idx <- c(1:nrow(train.cv))
      data.gbm$test.idx <-  c(1:nrow(test.cv))
      data.gbm$data.tr <- train.cv
      data.gbm$data.test <- test.cv
      data.gbm$data.type <- "te" 
      n.trees0 <- 4000
    }
    
    library(gbm)
    
    model.gbm <- gbm(
      action~.,
      data = data.gbm$data.tr[,flist],
      distribution = "bernoulli",
      n.trees = n.trees0,
      shrinkage = 0.03,
      interaction.depth = 5,
      train.fraction = 1.0,
      bag.fraction = 0.5,
      n.minobsinnode = 10,
      keep.data = T,
      cv.folds = 0,
      verbose = F)
    
    if (k <= data.cv.folds.cv$K) {
      library(Metrics)
      for (pred.trees in seq(100,4000,100)) {
        pred <- predict(model.gbm,
                        data.gbm$data.test[,flist], 
                        n.trees = pred.trees,
                        type = "response")
        print (paste("ntree =",pred.trees,": error =",auc(data.gbm$data.test[,'action'],pred)))
        model.gbm <- gbm.more(model.gbm,n.new.trees=100)
      }
      print(summary(model.gbm, plotit = F))
      pred.test.single <- predict(model.gbm,
                                  test.cv[,flist], 
                                  n.trees = 4000,
                                  type = "response")
    } else {
      pred <- predict(model.gbm,
                      data.gbm$data.test[,flist], 
                      n.trees = n.trees0,
                      type = "response")
      pred.test.single <- pred
    }
    fn.clean.worker()
    list(pred, pred.test.single)
  }
  fn.kill.wk()
  
  #extract prediction
  #on train (for 1 fold)
  for (k in 1:data.cv.folds.cv$K) {
    pred.gbm.train.hom[which(data.cv.folds$which==iter)] <- pred.gbm.train.hom[which(data.cv.folds$which==iter)] + gbm.pred[[k]][[2]]
  }
  print (auc(test.cv[,'action'],pred.gbm.train.hom[which(data.cv.folds$which==iter)]))
}
#extract prediction
#on train (homogeneous)
resdf <- data.frame(id=train.lb$id,action=action0,similarity=pred.gbm.train.hom/data.cv.folds$K,libfm=train.lb$libfm)
resdf[,"pred"] <- resdf$libfm
ix <- which(resdf$libfm<=0.9)
resdf[ix,"pred"] <- 1-resdf$similarity[ix]
ix <- which(resdf$libfm>0.9)
resdf[ix,"pred"] <- resdf$similarity[ix]
print (auc(resdf$action,resdf$pred))
resdf <- resdf[order(resdf$id),]
pred.gbm.train.hom <- resdf[,c("id","pred")]
print (auc(action0,pred.gbm.train.hom[,"pred"]))
pred.train <- pred.gbm.train.hom[,"pred",drop=FALSE]

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))