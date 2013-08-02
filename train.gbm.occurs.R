source("fn.base.R")
library("data.table")

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
data.cv.folds <- fn.cv.folds(nrow(data.tr), K = 100, seed = 3764743)
cat("done \n")
toc()

cat("Building previous occurrences table...\n")
fn.register.wk()
prev.occurs <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  library("data.table")
  val.select <-  fn.cv.which(data.cv.folds, k)
  val.idx <- which(val.select)
  tr.idx <- which(!val.select)
    
  if (k <= data.cv.folds$K) {
    data.tr.occurs <- data.tr[tr.idx,]
    data.test.occurs <- data.tr[val.idx,]
    data.test.occurs$test.idx <- val.idx
    data.test.occurs$datatype <- "tr"
  } else if (k == (data.cv.folds$K+1)) {
    data.tr.occurs <- data.tr
    data.test.occurs <- data.test
    data.test.occurs$test.idx <- 1:nrow(data.test.occurs)
    data.test.occurs$datatype <- "test"
  }
  
  data.test.occurs$ACTION <- NULL
  data.test.occurs.key <- data.test.occurs
  
  cols.occurs <- colnames(data.tr)[-1]
  cols.n <- length(cols.occurs)
  
  for (comb.n in c(1,2,3,6,7)) {
    grid.occurs <- combn(cols.n,comb.n)
    
    for (ci in 1:ncol(grid.occurs)) {
      col.names <- cols.occurs[grid.occurs[,ci]]
      skip <- all(c("ROLE_CODE","ROLE_FAMILY") %in% col.names)
      if (!skip) {
        col.occur.name <- paste(col.names, collapse="_")
        data.tr.occurs.dt <- data.table(data.tr.occurs)[
          ,list(n=length(ACTION)), by=col.names]
        setkeyv(data.tr.occurs.dt, col.names)
        if (sum(data.tr.occurs.dt$n > 1) > 0) {
          data.test.occurs[[col.occur.name]] <- data.tr.occurs.dt[
            J(data.test.occurs.key[,col.names])]$n
          data.test.occurs[[col.occur.name]][is.na(data.test.occurs[[col.occur.name]])] <- 0
        }
      }
    }
  }
  data.test.occurs
}
fn.kill.wk()
data.tr.occurs <- fn.extract.pred(prev.occurs, "tr2")
print(summary(data.tr.occurs$RESOURCE))
data.test.occurs <- fn.extract.pred(prev.occurs, "test2")
print(summary(data.test.occurs$RESOURCE))

fn.save.data("data.tr.occurs")
fn.save.data("data.test.occurs")

#############################################################
# train using gbm.occurs
#############################################################
data.tr <- cbind(data.tr[,"ACTION", drop = F], data.tr.occurs)
data.test <- cbind(data.test[,"ACTION", drop = F], data.test.occurs)

model.load <- F
model.pred.trees <- c(1000, 2000, 3000)

fn.register.wk()
gbm.occurs.pred <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  
  data.gbm.occurs <- list()
  
  val.select <-  fn.cv.which(data.cv.folds, k)
  
  data.gbm.occurs$log <- paste0("gbm_occurs_",k)
  data.gbm.occurs$log.full <- paste0("/log/",data.gbm.occurs$log, ".log")
  
  data.gbm.occurs$tr.idx <- which(!val.select)
  data.gbm.occurs$tr.x <- data.tr[data.gbm.occurs$tr.idx,-1]
  data.gbm.occurs$tr.y <- data.tr$ACTION[data.gbm.occurs$tr.idx]
  data.gbm.occurs$val.idx <- which(val.select)
  data.gbm.occurs$val.x <- data.tr[data.gbm.occurs$val.idx,-1]
  data.gbm.occurs$val.y <- data.tr$ACTION[data.gbm.occurs$val.idx]
  data.gbm.occurs$test.x <- data.test[,-1]
  
  library("gbm")
  
  model.name <- paste0("model.gbm.occurs.",k)
  model.file <- fn.data.file(paste0(model.name, ".RData"))
  model.load <- fn.get("model.load",F)
  
  if (model.load) {
    
    load(file = model.file)
    
  } else {

    model.gbm = gbm.fit(
           x = rbind(data.gbm.occurs$tr.x, data.gbm.occurs$val.x),
           y = c(data.gbm.occurs$tr.y, data.gbm.occurs$val.y),
           distribution = "bernoulli",
           n.trees = 3000,
           shrinkage = 0.05,
           nTrain = nrow(data.gbm.occurs$tr.x),
           interaction.depth = 20) 
    
    
    save(model.gbm, file = model.file)
    print(model.gbm)
    print(summary(model.gbm, plotit=F))
  }
  
  pred.trees <- model.gbm$n.trees
  model.pred.trees <- fn.get('model.pred.trees', NULL)
  if (model.load && !is.null(model.pred.trees)) {
    pred.trees <- model.pred.trees
  }
  
  data.pred <- NULL
  if (NROW(data.gbm.occurs$val.x) > 0) {
    
    data.pred <- data.frame(
      datatype = "tr",
      test.idx = data.gbm.occurs$val.idx,
      pred = 0)
    
    for (n.trees in pred.trees) {
      data.pred$pred <- data.pred$pred +
        predict(
          model.gbm, data.gbm.occurs$val.x, 
          n.trees,
          type="response")
      fn.print.auc.err(data.gbm.occurs$val.y,  data.pred$pred)
    }
    data.pred$pred <- data.pred$pred/length(pred.trees)
  }
  
  data.pred.test <- data.frame(
    datatype = "test",
    test.idx = 1:nrow(data.test),
    pred = 0)

  for (n.trees in pred.trees) {
    data.pred.test$pred <- data.pred.test$pred + 
      predict(
        model.gbm, data.gbm.occurs$test.x, 
        n.trees,
        type="response")
  }
  data.pred.test$pred <- data.pred.test$pred/length(pred.trees)
  
  rm(model.gbm)
  #fn.clean.worker()
  
  rbind(data.pred, data.pred.test)
}
fn.kill.wk()
#fn.save.data("gbm.occurs.pred")
# fn.load.data("gbm.occurs.pred")

#############################################################
# extract predictionl
# #############################################################
# fn.load.data("data.tr.gbm.occurs")
# fn.load.data("data.test.gbm.occurs")

data.tr.gbm.occurs <- fn.extract.tr(gbm.occurs.pred)

fn.print.auc.err(data.tr, data.tr.gbm.occurs)
#   Length       AUC
# 1  32769 0.8964196

data.test.gbm.occurs <- fn.extract.test(gbm.occurs.pred)
print(summary(data.tr.gbm.occurs))
print(summary(data.test.gbm.occurs))
