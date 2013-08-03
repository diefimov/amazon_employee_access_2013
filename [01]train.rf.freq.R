#setwd("D:/Dropbox/Eclipse/Amazon")
source("fn.base.R")
n.folds <- 10
alg.name <- "rf_freq"

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

#person
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

###################################################################
###### CV AND PREDICTION ##########################################
###################################################################

train.lb <- data.frame(tt[action>=0])
test.lb <- data.frame(tt[action<0])
train.lb[is.na(train.lb)] <- -1
test.lb[is.na(test.lb)] <- -1

set.seed(3847569)
data.cv.folds <- cvFolds(nrow(train.lb), K = n.folds, type="interleaved")
cat("Instance CV distribution: \n")
print(table(data.cv.folds$which))

###################################################################
###### RANDOM FOREST ##############################################
###################################################################

flist <- setdiff(colnames(train.lb),c("id","action"))
fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
rf.pred <- foreach(k=1:(data.cv.folds$K+1),.errorhandling="remove") %dopar% {
  file.name <- "output_rf"
  fn.init.worker(paste(file.name,k,sep=""))

  if (k <= data.cv.folds$K) {
    data.rf <- list()
    data.rf$tr.idx <- which(data.cv.folds$which!=k)
    data.rf$test.idx <-  which(data.cv.folds$which==k)
    data.rf$data.tr <- train.lb[data.rf$tr.idx,]
    data.rf$data.test <- train.lb[data.rf$test.idx,]
    data.rf$data.type <- "tr" 
  } else {
    data.rf <- list()
    data.rf$tr.idx <- c(1:nrow(train.lb))
    data.rf$test.idx <-  c(1:nrow(test.lb))
    data.rf$data.tr <- train.lb
    data.rf$data.test <- test.lb
    data.rf$data.type <- "te" 
  }
  
  library(randomForest)

  set.seed(543)
  model.rf <- randomForest(
    x = data.rf$data.tr[,flist],
    y = data.rf$data.tr[,"action"],
    ntree = 1000,
    mtry = 1,
    nodesize = 7)
  
  if (k <= data.cv.folds$K) {
    library(Metrics)
    pred <- predict(model.rf, data.rf$data.test[,flist], type="response")
    print (paste("error =",auc(data.rf$data.test[,'action'],pred)))
  } else {
    pred <- predict(model.rf, data.rf$data.test[,flist], type="response")
  }
  fn.clean.worker()
  pred
}
fn.kill.wk()

#extract prediction
#on train
pred.rf.train <- rep(0,nrow(train.lb))
for (k in 1:data.cv.folds$K) {
  pred.rf.train[which(data.cv.folds$which==k)] <- rf.pred[[k]]
}
print (auc(train.lb[,'action'],pred.rf.train))
pred.rf.train <- data.frame(id = train.lb$id, pred = pred.rf.train)
pred.rf.train <- pred.rf.train[order(pred.rf.train$id),]
pred.train <- pred.rf.train[,"pred",drop=FALSE]

#on test (single)
pred.rf.test <- data.frame(id = test.lb$id, pred = rf.pred[[data.cv.folds$K+1]])
pred.rf.test <- pred.rf.test[order(pred.rf.test$id),]
pred.test <- pred.rf.test[,"pred",drop=FALSE]

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))