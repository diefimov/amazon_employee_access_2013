source("fn.base.R")
n.folds <- 10
alg.name <- "libfm"

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

#############################################################
# train using libfm
#############################################################
fn.register.wk()
libfm.pred <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  
  data.libfm <- list()
  
  val.select <-  fn.cv.which(data.cv.folds, k)
  
  data.libfm$log <- paste0("libfm_",k)
  data.libfm$log.full <- paste0("log/",data.libfm$log, ".log")
  
  data.tr.libfm <- data.tr
  data.tr.libfm$ROLE_FAMILY <- NULL
  data.test.libfm <- data.test
  data.test.libfm$ROLE_FAMILY <- NULL
  
  fn.init.worker(data.libfm$log)
  
  data.libfm$tr.idx <- which(!val.select)
  data.libfm$tr <- data.tr.libfm[data.libfm$tr.idx,]
  data.libfm$val.idx <- which(val.select)
  data.libfm$val <- data.tr.libfm[data.libfm$val.idx,]
  data.libfm$test <- data.test.libfm
  data.libfm$name <- paste0("data.libfm.",k)
  data.libfm$iters <- 10
  
  fn.write.libsvm(
    data.tr = data.libfm$tr,
    data.test = rbind(data.libfm$val,data.libfm$test),
    name = data.libfm$name,
    dir = "libfm"
  )

  disk <- unlist(strsplit(path.wd,"/"))[1]
  path <- paste0(path.wd,"/libfm")
  for (it in 1:data.libfm$iters) {
	  lib_command <- paste0("libFM -train ", data.libfm$name, ".tr.libsvm -test ", data.libfm$name, ".test.libsvm -out ", data.libfm$name, ".test.pred.", it, " -init_stdev 0.5 -method mcmc -dim '1,1,12' -task c -iter 1500")
	  shell(paste(disk, "&& cd", path, "&&", lib_command, ">> ", data.libfm$log.full),translate=TRUE)
  }
  
  data.libfm.out <- NULL
  for (i in 1:data.libfm$iters) {
    data.libfm.cur <- read.csv(
      file = paste0("libfm/",data.libfm$name, ".test.pred.", i),
      header = F)$V1
    if (i == 1) {
      data.libfm.out <- data.libfm.cur
    } else {
      data.libfm.out <- data.libfm.out + data.libfm.cur
    }
  }
  data.libfm.out <- data.libfm.out/data.libfm$iters
  
  data.pred <- NULL
  if (NROW(data.libfm$val) > 0) {
  
    data.pred <- data.frame(
      datatype = "tr",
      test.idx = data.libfm$val.idx,
      pred = data.libfm.out[1:length(data.libfm$val.idx)])
    
    fn.print.auc.err(data.libfm$val$ACTION,  data.pred$pred)
    print(summary(data.pred$pred))
  }
  
  data.pred.test <- data.frame(
      datatype = "test",
      test.idx = 1:nrow(data.test),
      pred = tail(data.libfm.out, n = nrow(data.test)))
  
  print(summary(data.pred.test$pred))
  
  fn.clean.worker()

  
  rbind(data.pred, data.pred.test)
}
fn.kill.wk()

#############################################################
# extract predictionl
#############################################################
pred.train <- fn.extract.tr(libfm.pred)
fn.print.auc.err(data.tr, pred.train)
#   Length       AUC
# 1  32769 0.8936904
print(summary(pred.train))

pred.test <- fn.extract.test(libfm.pred)
print(summary(pred.test))

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))