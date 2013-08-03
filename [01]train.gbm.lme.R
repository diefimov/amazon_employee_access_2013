source("fn.base.R")
library("data.table")
n.folds <- 10
alg.name <- "gbm_lme"

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

cat("creating likelihood stats features... ")
fn.register.wk()
data.lme.likelihood <- foreach(k=1:(data.cv.folds$K+1)) %dopar% {

  fn.init.worker(paste("data_stats_",k,sep=""))
  
  library("data.table")
  library("Metrics")
  
  cv.select <- fn.cv.which(data.cv.folds, k)
  data.stats <- list()
  data.stats$tr.idx <- which(!cv.select)
  data.stats$tr <- data.tr[data.stats$tr.idx,]
  data.stats$cv.folds <- fn.cv.folds(nrow(data.stats$tr), 
                                       K = n.folds, 
                                       seed = 3764743)
  
  data.stats$tr.lme <- data.stats$tr[,0]
  
  data.stats$val.idx <- NULL
  data.stats$test <- NULL
  if (k <= data.cv.folds$K) {
    data.stats$val.idx <- which(cv.select)
    data.stats$test <- data.tr[data.stats$val.idx,]  
  } 
  
  data.stats$test <- rbind(data.stats$test,
                           data.test)
  
  data.stats$test.lme <- data.stats$test[,0]
  data.stats$order.lme <- data.stats$test[1,0]
  data.stats$auc.lme <- data.stats$test[1,0]
  
  cols.lme <- colnames(data.tr)[-1]
  cols.n <- length(cols.lme)
  
  for (comb.n in c(1,2,3,6,7)) {
    
    cat("Training ", comb.n, " order models... \n")
    
    grid.lme <- combn(cols.n,comb.n)
    
    for (c in 1:ncol(grid.lme)) {
      
      cols.stats <- cols.lme[grid.lme[,c]]
      skip <- all(c("ROLE_CODE","ROLE_FAMILY") %in% cols.stats)
      if (!skip) {
      
        cols.stats.join <- paste(cols.stats, collapse = "_")
        
        cat("Training ", cols.stats.join, "likelihood model... \n")
        
        stats.test <- fn.build.stats(cols.stats,
                                     data.stats$tr,
                                     data.stats$test)
        
        stats.tr <- fn.build.stats.cv(cols.stats,
                                      data.stats$tr,
                                      data.stats$cv.folds)
        
        if (any(stats.tr$n > 0) && any(stats.test$n > 0)) {
          
          stats.likelihood <- fn.lme.likelihood(
            rbind(stats.tr,stats.test))
          
          data.stats$tr.lme[[cols.stats.join]] <- 
            stats.likelihood$pred[1:nrow(stats.tr)]
          data.stats$test.lme[[cols.stats.join]] <- 
            stats.likelihood$pred[-(1:nrow(stats.tr))]
          data.stats$order.lme[[cols.stats.join]] <- comb.n
          
          prob.auc.tr <- auc(data.stats$tr$ACTION, 
                             data.stats$tr.lme[[cols.stats.join]])
          cat(cols.stats.join, "TR AUC: ", prob.auc.tr, "\n")
          
          val.action <- !is.na(data.stats$test$ACTION)
          if (any(val.action)) {
            prob.auc.val <- 
              auc(data.stats$test$ACTION[val.action], 
                  data.stats$test.lme[[cols.stats.join]][val.action])
            cat(cols.stats.join, "VAL AUC: ", prob.auc.val, "\n")
          } 
          
          if (prob.auc.tr <= 0.5) {
            data.stats$tr.lme[[cols.stats.join]] <- NULL
            data.stats$test.lme[[cols.stats.join]] <- NULL
            data.stats$order.lme[[cols.stats.join]] <- NULL
            cat("Discarded ", cols.stats.join, " \n")
          }
        } else {
          cat("Skipped ", cols.stats.join, " \n")
        }
        cat("\n")
      }
    }
  }
  cat("\n\n")
  
  if (length(data.stats$val.idx) > 0) {
    data.stats$tr.lme <- 
      rbind(data.stats$tr.lme,
            data.stats$test.lme[1:length(data.stats$val.idx),])
    data.stats$test.lme <- 
      data.stats$test.lme[-c(1:length(data.stats$val.idx)),]
  } 
  
  data.stats$tr.lme[c(data.stats$tr.idx,
                      data.stats$val.idx),] <- data.stats$tr.lme
  
  for (col.name in colnames(data.stats$tr.lme)) {
    if (!(col.name %in% c("datatype", "k"))) {
      auc.score <- auc(data.tr$ACTION, data.stats$tr.lme[[col.name]])
      cat(col.name, "OVERALL AUC: ", auc.score, "\n")
      data.stats$auc.lme[col.name] <- auc.score
    }
  }
  
  lme.name <- paste0("data.lme.likelihood.all.cv", data.cv.folds$K, ".",k)
  assign(lme.name, list(tr = data.stats$tr.lme,
                        test = data.stats$test.lme,
                        order = data.stats$order.lme,
                        auc = data.stats$auc.lme))
  fn.save.data(lme.name)
  
  fn.clean.worker()
  
  NULL
}
fn.kill.wk()

#############################################################
# train using lme
#############################################################
model.load <- F
model.pred.trees <- c(1000, 2000, 3000)

fn.register.wk()
gbm.lme.pred <- foreach(k=1:(data.cv.folds$K+1),.combine=rbind) %dopar% {
  
  fn.init.worker(paste("gbm_lme_",k,sep=""))
  
  lme.name <- paste0("data.lme.likelihood.all.cv", data.cv.folds$K, ".", k)
  fn.load.data(lme.name)
  data.lme.likelihood <- get(lme.name)
  
  data.lme <- list()
  data.lme$cols <- colnames(data.lme.likelihood$order)
  cols.ord <- t(data.lme.likelihood$order[1,])[,]
  data.lme$cols <- names(cols.ord)#[cols.ord <= 3 | cols.ord >= 5]


  data.tr.lme <- 
    cbind(data.tr[,"ACTION", drop = F],
          data.lme.likelihood$tr)
  
  data.test.lme <- 
    cbind(data.test[,"ACTION", drop = F],
          data.lme.likelihood$test)
  
  cv.select <- fn.cv.which(data.cv.folds, k)
  data.lme$tr.idx  <- which(!cv.select)
  data.lme$tr <- data.tr.lme[data.lme$tr.idx,]
  data.lme$val.idx <- which(cv.select)
  data.lme$val<- data.tr.lme[data.lme$val.idx,]
  data.lme$test <- data.test.lme
  
  data.lme$tr <- rbind(data.lme$tr,data.lme$val)
  
  model.name <- paste0("model.gbm.lme.",k)
  model.file <- fn.data.file(paste0(model.name, ".RData"))
  model.load <- fn.get("model.load",F)
  
  library("gbm")
  if (model.load) {
    
    load(file = model.file)
    data.lme$cols <- model.gbm$var.names
  } else {
    
    model.gbm <- gbm.fit(
      x = data.lme$tr[,data.lme$cols],
      y = data.lme$tr$ACTION,
      distribution = "bernoulli",
      nTrain = length(data.lme$tr.idx),
      n.trees = 3000,
      shrinkage = 0.005,
      interaction.depth = 30,
      keep.data = F,
      verbose = T)
    
    #save(model.gbm, file = model.file)
    
    print(model.gbm)
    print(summary(model.gbm, plotit=F))
  }
  
  pred.trees <- model.gbm$n.trees
  model.pred.trees <- fn.get('model.pred.trees', NULL)
  if (model.load && !is.null(model.pred.trees)) {
    pred.trees <- model.pred.trees
  }
  
  data.pred <- NULL
  if (NROW(data.lme$val) > 0) {
    
    data.pred <- data.frame(
      datatype = "tr",
      test.idx = data.lme$val.idx,
      pred = 0)
    
    for (n.trees in pred.trees) {
      data.pred$pred <- data.pred$pred +
        predict(
          model.gbm, data.lme$val[,data.lme$cols], 
          n.trees,
          type="response")
      fn.print.auc.err(data.lme$val$ACTION,  data.pred$pred)
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
        model.gbm, data.lme$test[,data.lme$cols], 
        n.trees,
        type="response")
  }
  data.pred.test$pred <- data.pred.test$pred/length(pred.trees)

  fn.clean.worker()
  
  rbind(data.pred, data.pred.test)
}
fn.kill.wk()

# #############################################################
# # extract predictions
# #############################################################

pred.train <- fn.extract.tr(gbm.lme.pred)
fn.print.auc.err(data.tr, pred.train)
#   Length      AUC
# 1  32769 0.907547

pred.test <- fn.extract.test(gbm.lme.pred)
print(summary(pred.train))
print(summary(pred.test))

save(pred.test, pred.train, file=paste0("output-R/",alg.name,".RData"))
