#setwd("D:/Dropbox/Eclipse/Amazon")
source("utils.R")

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

#for persons
tt1 <- tt[,list(resource,person)]
tt2 <- tt[,list(resource,person)]
tt.merge <- merge(tt1,tt2,by="resource",allow.cartesian=TRUE)
setnames(tt.merge,c("resource","person1","person2"))
tt.merge <- tt.merge[person1!=person2]
tt.persons <- tt.merge[,list(person1,person2)]
setkey(tt.persons,person1,person2)
tt.persons <- unique(tt.persons)
tt.persons.features <- tt[,list(person,mgr,role1,role2,dept,title,desc,fam)]
setkey(tt.persons.features,person)
tt.persons.features <- unique(tt.persons.features)
setnames(tt.persons.features,"person","person1")
tt.persons <- merge(tt.persons,tt.persons.features,by="person1",all.x=TRUE)
setnames(tt.persons.features,"person1","person2")
tt.persons <- merge(tt.persons,tt.persons.features,by="person2",all.x=TRUE)
tt.persons[,mgr:=ifelse(mgr.x==mgr.y,1,0)]
tt.persons[,role1:=ifelse(role1.x==role1.y,1,0)]
tt.persons[,role2:=ifelse(role2.x==role2.y,1,0)]
tt.persons[,dept:=ifelse(dept.x==dept.y,1,0)]
tt.persons[,title:=ifelse(title.x==title.y,1,0)]
tt.persons[,desc:=ifelse(desc.x==desc.y,1,0)]
tt.persons[,fam:=ifelse(fam.x==fam.y,1,0)]
tt.persons <- tt.persons[,list(person1,person2,mgr,role1,role2,dept,title,desc,fam)]
#distance by mgr, role2, dept
tt.persons[,dist1:=mgr+role2+dept]
#merging with initial data
tt.merge <- merge(tt.merge,tt.persons[,list(person1,person2,dist1)],by=c("person1","person2"),all.x=TRUE)
tt.person.closest <- tt.merge[,list(dist1=max(dist1)),by=c("person1","resource")]
setnames(tt.person.closest,"person1","person")
tt <- merge(tt,tt.person.closest,by=c("person","resource"),all.x=TRUE)
tt[,dist1:=ifelse(is.na(dist1),0,dist1)]

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

###################################################################
###### CV AND PREDICTION ##########################################
###################################################################

train.lb <- data.frame(tt[action>=0])
test.lb <- data.frame(tt[action<0])
train.lb[is.na(train.lb)] <- -1
test.lb[is.na(test.lb)] <- -1

set.seed(3847569)
data.cv.folds <- cvFolds(nrow(train.lb), K = 100, type="interleaved")
cat("Instance CV distribution: \n")
print(table(data.cv.folds$which))

###################################################################
###### GBM ########################################################
###################################################################

flist <- setdiff(colnames(train.lb),c("id"))
cols.cat <- c("resource","dept","person","role2","title","desc","mgr","uperson","role3")
fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
gbm.pred <- foreach(k=1:(data.cv.folds$K+1),.errorhandling="remove") %dopar% {
  file.name <- "output_gbm3"
  fn.init.worker(paste(file.name,k,sep=""))
  
  tt.coding <- rbind(train.lb,test.lb)
  for (col in cols.cat) {
    tt.coding[, col] <- assign_random_values(tt.coding[, col])
  }
  train.coding <- tt.coding[1:nrow(train.lb),]
  test.coding <- tt.coding[-(1:nrow(train.lb)),]
  
  if (k <= data.cv.folds$K) {
    data.gbm <- list()
    data.gbm$tr.idx <- which(data.cv.folds$which!=k)
    data.gbm$test.idx <-  which(data.cv.folds$which==k)
    data.gbm$data.tr <- train.coding[data.gbm$tr.idx,]
    data.gbm$data.test <- train.coding[data.gbm$test.idx,]
    data.gbm$data.type <- "tr" 
    n.trees0 <- 100
  } else {
    data.gbm <- list()
    data.gbm$tr.idx <- c(1:nrow(train.lb))
    data.gbm$test.idx <-  c(1:nrow(test.lb))
    data.gbm$data.tr <- train.coding
    data.gbm$data.test <- test.coding
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
                                test.coding[,flist], 
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
pred.gbm.train <- data.frame(id = train.lb$id, gbm_freq3 = pred.gbm.train)
pred.gbm.train <- pred.gbm.train[order(pred.gbm.train$id),]

#on test (homogeneous)
pred.gbm.test.hom <- rep(0,nrow(test.lb))
for (k in 1:data.cv.folds$K) {
  pred.gbm.test.hom <- pred.gbm.test.hom + gbm.pred[[k]][[2]]
}
pred.gbm.test.hom <- data.frame(id = test.lb$id, gbm_freq_hom3_100 = pred.gbm.test.hom/data.cv.folds$K)
pred.gbm.test.hom <- pred.gbm.test.hom[order(pred.gbm.test.hom$id),]

#homogeneous gbm on train
pred.gbm.train.hom <- rep(0,nrow(train.lb))
for (iter in 1:data.cv.folds$K) {
  print (paste("iter:",iter))
  ix.prev.cv <- which(data.cv.folds$which==iter)
  
  train.cv <- train.lb[which(data.cv.folds$which!=iter),]
  test.cv <- train.lb[which(data.cv.folds$which==iter),]
  data.cv.folds.cv <- cvFolds(nrow(train.cv), K = 100)
  cat("Instance CV distribution: \n")
  print(table(data.cv.folds.cv$which))
  
  flist <- setdiff(colnames(train.cv),c("id"))
  cols.cat <- c("resource","dept","person","role2","title","desc","mgr","uperson","role3")
  fn.register.wk(as.numeric(Sys.getenv('NUMBER_OF_PROCESSORS'))-1)
  gbm.pred <- foreach(k=1:(data.cv.folds.cv$K),.errorhandling="remove") %dopar% {
    file.name <- "output_gbm3"
    fn.init.worker(paste(file.name,k,sep=""))
    
    tt.coding <- rbind(train.cv,test.cv)
    for (col in cols.cat) {
      tt.coding[, col] <- assign_random_values(tt.coding[, col])
    }
    train.coding <- tt.coding[1:nrow(train.cv),]
    test.coding <- tt.coding[-(1:nrow(train.cv)),]
    
    if (k <= data.cv.folds.cv$K) {
      data.gbm <- list()
      data.gbm$tr.idx <- which(data.cv.folds.cv$which!=k)
      data.gbm$test.idx <-  which(data.cv.folds.cv$which==k)
      data.gbm$data.tr <- train.coding[data.gbm$tr.idx,]
      data.gbm$data.test <- train.coding[data.gbm$test.idx,]
      data.gbm$data.type <- "tr" 
      n.trees0 <- 100
    } else {
      data.gbm <- list()
      data.gbm$tr.idx <- c(1:nrow(train.cv))
      data.gbm$test.idx <-  c(1:nrow(test.cv))
      data.gbm$data.tr <- train.coding
      data.gbm$data.test <- test.coding
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
                                  test.coding[,flist], 
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
    pred.gbm.train.hom[ix.prev.cv] <- pred.gbm.train.hom[ix.prev.cv] + gbm.pred[[k]][[2]]
  }
  print (auc(test.cv[,'action'],pred.gbm.train.hom[ix.prev.cv]))
}
#extract prediction
#on train (homogeneous)
pred.gbm.train.hom <- data.frame(id = train.lb$id, gbm_freq_hom3_100 = pred.gbm.train.hom/data.cv.folds.cv$K)
print (auc(train.lb[,'action'],pred.gbm.train.hom[,"gbm_freq_hom3_100"]))
pred.gbm.train.hom <- pred.gbm.train.hom[order(pred.gbm.train.hom$id),]
