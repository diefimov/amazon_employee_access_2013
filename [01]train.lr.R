source("fn.base.R")

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

#############################################################
# train using lr
#############################################################
fn.register.wk() # 1:(data.cv.folds$K+1)
lr.pred <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  
  data.lr <- list()
  
  val.select <-  fn.cv.which(data.cv.folds, k)
  
  data.lr$log <- paste0("lr_",k)
  data.lr$log.full <- paste0("../data/log/",data.lr$log, ".log")
  
  data.tr.lr <- data.tr
  data.test.lr <- data.test
  data.test.lr$ACTION <- -1
  
  fn.init.worker(data.lr$log)
  cat("Fold ", k, "\n")
  
  data.lr$tr.idx <- which(!val.select)
  data.lr$tr <- data.tr.lr[data.lr$tr.idx,]
  data.lr$val.idx <- which(val.select)
  data.lr$val <- data.tr.lr[data.lr$val.idx,]
  data.lr$test <- data.test.lr
  data.lr$name <- paste0("data.lr.",k)
  data.lr$iters <- 2
  
  data.lr.out <- NULL
  
  it.start <- 1
#   if (k %in% c(7)) { it.start <- 7 }
  for (it in it.start:data.lr$iters) {
    
    it.name <- paste0(data.lr$name, ".", it)
    tr.name <- paste0("data/lr/",it.name, ".tr.csv")
    test.name <- paste0("data/lr/",it.name, ".test.csv")
    test.pred.name <- paste0("data/lr/",it.name, ".test.pred.csv")
    write.csv(data.lr$tr,
              file = tr.name,
              row.names = F, quote = F)
    write.csv(rbind(data.lr$val, data.lr$test),
              file = test.name,
              row.names = F, quote = F)
    
    lr.seed <- sample(1e7,1)
	disk <- unlist(strsplit(path.wd,"/"))[1]
    shell(paste(disk, "&& cd", path.wd, "&& python -u logistic_regression.py", 
              tr.name, test.name, test.pred.name, lr.seed, "3 >> ", data.lr$log.full),translate=TRUE)
    
    data.lr.cur <- read.csv(file = test.pred.name)$ACTION
    if (NROW(data.lr$val) > 0) {
      fn.print.auc.err(data.lr$val$ACTION,  data.lr.cur[1:length(data.lr$val.idx)])
    }
    if (is.null(data.lr.out)) {
      data.lr.out <- data.lr.cur
    } else {
      data.lr.out <- data.lr.out + data.lr.cur
    }
  }
  
  data.lr.out <- data.lr.out/data.lr$iters
  
  data.pred <- NULL
  if (NROW(data.lr$val) > 0) {
  
    data.pred <- data.frame(
      datatype = "tr",
      test.idx = data.lr$val.idx,
      pred = data.lr.out[1:length(data.lr$val.idx)])
    
    fn.print.auc.err(data.lr$val$ACTION,  data.pred$pred)
    print(summary(data.pred$pred))
  }
  
  data.pred.test <- data.frame(
      datatype = "test",
      test.idx = 1:nrow(data.test),
      pred = tail(data.lr.out, n = nrow(data.test)))
  
  print(summary(data.pred.test$pred))
  
  fn.clean.worker()
  rbind(data.pred, data.pred.test)
}
fn.kill.wk()

#############################################################
# extract predictions
#############################################################
data.tr.lr <- fn.extract.tr(lr.pred)
fn.print.auc.err(data.tr, data.tr.lr)
#   Length       AUC
# 1  32769 0.8906123

data.test.lr <- fn.extract.test(lr.pred)
print(summary(data.tr.lr))
print(summary(data.test.lr))

