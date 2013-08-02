require("compiler")
enableJIT(3) 
setCompilerOptions(suppressUndefined = T)
options(stringsAsFactors = FALSE)
options(max.print = 1000)

path.wd <- getwd()

library(data.table)
library(cvTools)
library(gbm)
library(Metrics)
library(glmnet)

#############################################################
# tic toc
#############################################################
tic <- function(gcFirst = TRUE, type=c("elapsed", "user.self", "sys.self")) {
  type <- match.arg(type)
  assign(".type", type, envir=baseenv())
  if(gcFirst) gc(FALSE)
  tic <- proc.time()[type]         
  assign(".tic", tic, envir=baseenv())
  invisible(tic)
}

toc <- function() {
  type <- get(".type", envir=baseenv())
  toc <- proc.time()[type]
  tic <- get(".tic", envir=baseenv())
  print(toc - tic)
  invisible(toc)
}

##############################################################
## Registers parallel workers
##############################################################
fn.register.wk <- function(n.proc = NULL) {
  if (file.exists("data/cluster.csv")) {
    cluster.conf <- read.csv("data/cluster.csv", 
                             stringsAsFactors = F,
                             comment.char = "#")
    n.proc <- NULL
    for (i in 1:nrow(cluster.conf)) {
      n.proc <- c(n.proc, 
                  rep(cluster.conf$host[i], 
                      cluster.conf$cores[i]))
    }
  }
  if (is.null(n.proc)) {
    n.proc = as.integer(Sys.getenv("NUMBER_OF_PROCESSORS"))
    if (is.na(n.proc)) {
      library(parallel)
      n.proc <-detectCores()
    }
  }
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (!exists(".pworkers", envir=baseenv()) || length(workers$.pworkers) == 0) {
    
    library(doSNOW)
    library(foreach)
    workers<-suppressWarnings(makeSOCKcluster(n.proc));
    suppressWarnings(registerDoSNOW(workers))
    clusterSetupRNG(workers, seed=5478557)
    assign(".pworkers", workers, envir=baseenv());
    
    tic()
    cat("Workers start time: ", format(Sys.time(), format = "%Y-%m-%d %H:%M:%S"), "\n")
  }
  invisible(workers);
}

##############################################################
## Kill parallel workers
##############################################################
fn.kill.wk <- function() {
  library("doSNOW")
  library("foreach")
  workers <- mget(".pworkers", envir=baseenv(), ifnotfound=list(NULL));
  if (exists(".pworkers", envir=baseenv()) && length(workers$.pworkers) != 0) {
    stopCluster(workers$.pworkers);
    assign(".pworkers", NULL, envir=baseenv());
    cat("Workers finish time: ", format(Sys.time(), format = "%Y-%m-%d %H:%M:%S"), "\n")
    toc()
  }
  invisible(workers);
}

##############################################################
## init worker setting work dir and doing path redirect
##############################################################
fn.init.worker <- function(log = NULL, add.date = F) {
  setwd(path.wd)
  
  if (!is.null(log)) {
    date.str <- format(Sys.time(), format = "%Y-%m-%d_%H-%M-%S")
    
    if (add.date) {
      output.file <- fn.log.file(paste(log, "_",date.str,
                                       ".log", sep=""))
    } else {
      output.file <- fn.log.file(paste(log,".log", sep=""))
    }
    output.file <- file(output.file, open = "wt")
    sink(output.file)
    sink(output.file, type = "message")
    
    cat("Start:", date.str, "\n")
  }
  
  tic()
}

##############################################################
## clean worker resources
##############################################################
fn.clean.worker <- function() {
  gc()
  
  try(toc(), silent=T)
  suppressWarnings(sink())
  suppressWarnings(sink(type = "message"))
}

#############################################################
# log file path
#############################################################
fn.base.dir <- function(extra) {
  paste0(path.wd, "/", extra)
}

#############################################################
# log file path
#############################################################
fn.log.file <- function(name) {
  fn.base.dir(paste0("log/", name))
}

#############################################################
# input file path
#############################################################
fn.in.file <- function(name) {
  fn.base.dir(paste0("data/", name))
}

#############################################################
# python output file path
#############################################################
fn.out.file <- function(name) {
  fn.base.dir(paste0("output-R/", name))
}

#############################################################
# submission file path
#############################################################
fn.submission.file <- function(name) {
  fn.base.dir(paste0("submission/", name, ".csv"))
}

#############################################################
# data file path
#############################################################
fn.data.file <- function(name) {
  fn.out.file(name)
}

#############################################################
# save data file
#############################################################
fn.save.data <- function(dt.name, envir = parent.frame()) {
  save(list = dt.name, 
       file = fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}

#############################################################
# load saved file
#############################################################
fn.load.data <- function(dt.name, envir = parent.frame()) {
  load(fn.data.file(paste0(dt.name, ".RData")), envir = envir)
}

#############################################################
# error evaluation
#############################################################
fn.print.auc.err <- function(actual, pred, do.print = T) { 
  
  if (is.data.frame(actual)) {
    if (!is.null(actual$action)) {
      actual <- actual$action
    }  else if (!is.null(actual$ACTION)) {
      actual <- actual$ACTION
    }
  }
  
  if (is.data.frame(pred)) {
    pred <- pred$pred
  }
  
  if (is.null(actual) || all(is.na(actual))) {
    df <- summary(pred)
  } else {
    library("Metrics")
    df <- data.frame(Length = length(actual),
                     AUC = auc(actual,pred))
  }
  
  
  if (do.print) {
    print(df)
  }
  
  invisible(df)
}
# debug(fn.print.auc.err)


#############################################################
# join data tables - overwrite existing cols
#############################################################
fn.join.dt <- function(df1, df2) {
  cols.key <- key(df2)
  if (is.data.table(df1)) {
    data.key <- df1[,cols.key,with=F]
    df1 <- data.table(df1, key = key(df1))
  } else {
    data.key <- data.table(df1[,cols.key,drop=F])
    df1 <- data.frame(df1)
  }
  df2.join <- df2[data.key,allow.cartesian=TRUE]
  df2.join <- df2.join[
    ,!(colnames(df2.join) %in% cols.key),with=F]
  cols.overwrite <- colnames(df2.join)[
    colnames(df2.join) %in% colnames(df1)]
  for (col.name in cols.overwrite) {
    df1[[col.name]] <- NULL
  }
  cbind(df1, df2.join)
}



##################################################
# write libsvm data
##################################################
fn.write.libsvm <- function(
  data.tr, 
  data.test, 
  name,
  fn.y.transf = NULL, 
  dir = "/data/output",
  col.y = "ACTION",
  col.x = colnames(data.tr)[!(colnames(data.tr) %in% c(col.y, col.qid))],
  col.qid = NULL,
  feat.start = 1,
  vw.mode = F,
  data.val = NULL,
  y.def = min(data.tr[[col.y]]))
{
  options(scipen=999)
  library("data.table")
  cat("Building feature map ...")
  tic()
  model.dts <- list()
  val.xi <- feat.start
  col.x.groups <- NULL
  feat.size <- 0
  for (i in (1:length(col.x))) {
    col <- col.x[i]
    if (is.factor(data.tr[[col]])) {
      
      col.ids.tr   <- as.factor(levels(data.tr[[col]]))
      model.dts[[col]] <- data.table(ID = col.ids.tr, key = "ID")
      if (!is.null(data.val)) {
        col.ids.val <- as.factor(levels(data.val[[col]]))
        model.dts[[col]] <- merge(model.dts[[col]], 
                                  data.table(ID = col.ids.val, key = "ID"))
      }
      if (!is.null(data.test)) {
        col.ids.test <- as.factor(levels(data.test[[col]]))
        model.dts[[col]] <- merge(model.dts[[col]], 
                                  data.table(ID = col.ids.test, key = "ID"))
      }
      feat.size <- feat.size + nrow(model.dts[[col]])
      
      model.dts[[col]]$X.Idx <- val.xi:(val.xi+nrow(model.dts[[col]])-1)
      
      val.xi <- val.xi+nrow(model.dts[[col]])
      
    } else {
      model.dts[[col]] <- val.xi
      val.xi <- val.xi + 1
    }
  }
  map.name <- paste(name, ".map", sep="")
  assign(map.name, model.dts)
  save(list = map.name, file = paste(dir, "/", map.name, ".RData", sep=""))
  cat("done \n")
  toc()
  
  if (!exists("cmpfun")) {
    cmpfun <- identity
  }
  write.file <- cmpfun(function (data, file) { 
    tic()
    col.chunk <- col.x
    if (!is.null(col.qid)) {
      col.chunk <- c(col.chunk, col.qid)
    }
    if (!is.null(data[[col.y]])) {
      col.chunk <- c(col.y, col.chunk)
    }
    unlink(file)
    cat("Saving ", file, "...")
    fileConn <- file(file, open="at")
    
    data.chunk <- data[, col.chunk]
    if (is.null(data.chunk[[col.y]])) {
      data.chunk[[col.y]] <- 0
    }
    data.chunk[[col.y]][is.na(data.chunk[[col.y]])] <- y.def
    if (!is.null(fn.y.transf)) {
      data.chunk[[col.y]] <- fn.y.transf(data.chunk[[col.y]])
    }
    data.chunk[[col.y]][data.chunk[[col.y]] == Inf] <- y.def
    
    for (col in col.x) {
      if (is.numeric(data.chunk[[col]])) {
        data.chunk[[col]][data.chunk[[col]] == 0] <- NA
      }
    }
    
    for (col in col.x) {
      if (is.factor(data.chunk[[col]])) {
        data.chunk[[col]] <-  paste(
          model.dts[[col]][J(data.chunk[[col]])]$X.Idx,
          c(1), sep = ":")
      } else {
        data.chunk[[col]] <- paste(
          rep(model.dts[[col]], nrow(data.chunk)),
          data.chunk[[col]], sep = ":")
      }
    }
    
    if (!is.null(col.qid)) {
      data.chunk[[col.qid]] <- paste("qid", data.chunk[[col.qid]], sep = ":")
    }
    
    data.chunk <- do.call(paste, data.chunk[, c(col.y, col.qid, col.x)])
    chunk.size <- as.numeric(object.size(data.chunk))
    chunk.size.ch <- T
    while (chunk.size.ch) {
      data.chunk <- gsub(" [0-9]+\\:?NA", "", data.chunk)
      data.chunk <- gsub(" NA\\:-?[0-9]+", "", data.chunk)
      chunk.size.ch <- chunk.size != as.numeric(object.size(data.chunk))
      chunk.size <- as.numeric(object.size(data.chunk))
    }
    data.chunk <- gsub("\\s+", " ", data.chunk)
    data.chunk <- gsub("^([0-9]+(\\.[0-9]+)?)\\s*$", paste0("\\1 ", val.xi, ":1"), data.chunk)
    
    if (vw.mode) {
      data.chunk <- gsub("^([-]?[0-9]+(\\.[0-9]+)?)\\s+", "\\1 | ", data.chunk)
    }
    
    writeLines(c(data.chunk), fileConn)
    
    close(fileConn)
    cat("done.\n")
    toc()
  })
#     debug(write.file)
  write.file(data.tr, paste(dir, "/", name, ".tr.libsvm", sep=""))
  if (!is.null(data.val)) {
    write.file(data.val, paste(dir, "/", name, ".val.libsvm", sep=""))
  }
  if (!is.null(data.test)) {
    write.file(data.test, paste(dir, "/", name, ".test.libsvm", sep=""))
  }
}
# debug(fn.write.libsvm)

##################################################
# expand factors
##################################################
fn.expand.factors <- function(data.df) {
  data.frame(model.matrix( ~ . - 1, data=data.df))
}

##############################################################
## get with default value
##############################################################
fn.get <- function(var.name, def.val = NULL) {
  if ( exists(var.name)) {
    return(get(var.name))
  } else {
    return(def.val)
  }
}

##############################################################
## calculates cross valid stats
##############################################################
fn.build.stats <- function(
  stats.cols,
  df.tr,
  df.test
  ) {
  
  library("data.table")

  data.stats.col <- data.table(df.tr)[
    ,list(pos = sum(ACTION == 1),
          n = length(ACTION)),
    by = c(stats.cols)]
  setkeyv(data.stats.col, stats.cols)
  
  df.test.new <- data.frame(df.test)
  df.test.new <- fn.join.dt(df.test.new, data.stats.col)
  df.test.new <- df.test.new[,colnames(data.stats.col)]
  df.test.new[is.na(df.test.new)] <- 0
  
  df.test.new
}
# debug(fn.build.stats)

##############################################################
## calculates cross valid stats
##############################################################
fn.build.stats.cv <- function(
  stats.cols,
  df.tr,
  cv.folds
  ) {
  
  df.stats <- NULL
  for (k in 1:cv.folds$K) {
    cv.select <- fn.cv.which(cv.folds, k)
    tr.idx <- which(!cv.select) 
    tr <- df.tr[tr.idx,]
    val.idx <- which(cv.select) 
    val <- df.tr[val.idx,]
    
    df.cur <- fn.build.stats(stats.cols,tr,val)
    df.cur$idx <- val.idx
    df.stats <- rbind(df.stats, df.cur)
  }
  df.stats[df.stats$idx,] <- df.stats
  df.stats$idx <- NULL
  df.stats
}
# debug(fn.build.stats.cv)

##############################################################
## calculates leave one out stats
##############################################################
fn.build.stats.loo <- function(
  stats.cols,
  df.tr
) {
  df.stats <- fn.build.stats(stats.cols,df.tr,df.tr)
  df.stats$n <- df.stats$n-1
  is.pos <- df.tr$ACTION == 1
  df.stats$pos[is.pos] <- df.stats$pos[is.pos]-1
  
  df.stats
}
# debug(fn.build.stats.loo)

##############################################################
## creates likelihood feature
##############################################################
fn.lme.likelihood <- function(
  data.stats,
  fn.weight.transf = identity,
  family = "gaussian",
  verbose = F) {
  
  library("data.table")
  
  feat.df <- unique(data.stats)
  feat.df <- feat.df[feat.df$n > 0,]
  feat.df$id <- 1:nrow(feat.df)
  
  if (family == "gaussian") {
    family <- gaussian()
    feat.lme <- feat.df
    feat.lme$target <- feat.lme$pos/feat.lme$n
    feat.lme$weights <- fn.weight.transf(feat.lme$n)
  }
  
  data.lme <- feat.lme[, c("target", "id", "weights")]
  data.lme <- rbind(data.lme,
                    data.frame(target = c(0,1),
                               id = -1,
                               weights = 1))
  lmer.model <- suppressWarnings(
      fn.train.lmer(formula=target ~ 1 + (1|id),
                    data = data.lme,
                    weights = weights,
                    verbose = verbose))
  
  data.stats.pred <- data.table(data.stats)
  feat.lme.id <- data.table(feat.lme[,c(colnames(data.stats), "id")], 
                            key = colnames(data.stats))
  data.stats.pred <- feat.lme.id[data.stats.pred]
  data.stats.pred$id[is.na(data.stats.pred$id)] <- -10
  data.stats.pred <- data.frame(data.stats.pred)
  data.stats.pred$pred <- lmer.model$predict(data.stats.pred)
  data.stats.pred$id <- NULL

  data.stats.pred
}
# debug(fn.lme.likelihood)

##############################################################
## Training lmer function function
##############################################################
fn.train.lmer <-  function(formula,
                           data,
                           family = binomial(),
                           weights = NULL,
                           verbose = T,
                           ...) {
  library("lme4")
  library("data.table")
  
  # model object
  model <- list();
  model$formula <- formula;
  model$family <- family;
  
    
  # Estimate LMER
  if (is.null(weights)) {
    lme.model = suppressWarnings(glmer(
      formula=formula,
      data=data,
      family = family,
      verbose = verbose,
#       optimizer = "Nelder_Mead",
      control = list(...)));
  } else {
    lme.model = suppressWarnings(glmer(
        formula=formula,
        data=data,
        family = family,
        weights = weights,
        verbose = verbose,
#         optimizer = "Nelder_Mead",
        control = list(...)));
  }
  
#   lme.model <- do.call("glmer", 
#                        list(formula=formula,
#                             data=data,
#                             family = family,
#                             weights = weights,
#                             verbose = verbose)) 
    
  # get the constant term
  model$fixef <- fixef(lme.model)
  model$ranef <- ranef(lme.model)
  
  model$const <- as.numeric(model$fixef)
  
  model$dt <- list();
  model.names <- names(model$ranef)
  for (name in model.names) {
    vals <- model$ranef[[name]];
    model$dt[[name]] <- data.table(
      as.numeric(rownames(vals)),
      vals[,])
    col.n <- NCOL(vals); 
    col.values <- paste(rep("V", col.n), c(1:col.n), sep = "")
    setnames(model$dt[[name]], c(name, col.values))
    setkeyv(model$dt[[name]], name)
  }
  gc()

  model$predict <- function(data) {
    library("data.table")
    pred <- rep(model$const, nrow(data))
    for (name in model.names) {
      pred.key <- data.table(data[[name]])
      pred.cur <- model$dt[[name]][pred.key]$V1
      pred.cur[is.na(pred.cur)] <- 0
      pred <- pred + pred.cur
    }
    return (model$family$linkinv(pred));
  }
#   debug(model$predict)
  
  invisible (model)
};
# debug(fn.train.lmer)

#############################################################
# extract prediction
#############################################################
fn.extract.pred <- function(data.all, data.type, try.join = ncol(data.all) == 3) {
  
  library("data.table")
  
  data.extracted <- data.all[data.all$datatype == data.type, ]
  data.extracted$datatype <- NULL
  data.extracted <- data.table(data.extracted)
  if (try.join) {
    data.extracted <- data.frame(data.extracted[
        ,data.frame(t(colMeans(.SD))), by="test.idx"])
  }
  data.extracted <- data.frame(data.extracted)
  
  data.extracted[data.extracted$test.idx,] <- data.extracted
  rownames(data.extracted) <- 1:nrow(data.extracted)
  
  cols.out <- colnames(data.extracted)[!(
    colnames(data.extracted) %in% c("test.idx", "datatype"))]
  
  data.extracted[, cols.out, drop = F]
}

# debug(fn.extract.pred)

fn.extract.tr <- function(data.all, ...) {
  fn.extract.pred(data.all, "tr", ...)
}

fn.extract.test <- function(data.all, ...) {
  fn.extract.pred(data.all, "test", ...)
}

#############################################################
# write submission
#############################################################
fn.write.submission <- function(pred, file.name) { 
  if (is.data.frame(pred)) {
    pred <- pred$pred
  }
  
  pred <- data.frame(id = 1:NROW(pred), ACTION=pred)
  write.csv(pred,
            file = fn.submission.file(paste0(file.name)), 
            row.names = F, quote = F)
}

##############################################################
## cross val folds
##############################################################
fn.cv.folds <- function(n, K, seed, type = "interleaved") {
  library("cvTools")

  set.seed(seed)
  data.cv.folds <- cvFolds(n, K = K, type = type)
  set.seed(Sys.time())
  
  data.cv.folds <- list(
    n = data.cv.folds$n,
    K = data.cv.folds$K,
    which = data.cv.folds$which,
    subsets = data.cv.folds$subsets)

  data.cv.folds
}

##############################################################
## cross val selection
##############################################################
fn.cv.which <- function(cv.data, k) {
  sel.idx <- cv.data$subsets[cv.data$which %in% k]
  sel.logical <- rep(F, cv.data$n)
  sel.logical[sel.idx] <- T
  sel.logical
}
# debug(fn.write.submission)
# ##############################################################
# ## tranform target of prediction
# ##############################################################
# fn.target.to.xor <- function(data.df, col, thres) {
#   thres.rng <- data.df[[col]] > thres
#   target <- data.df$target == 1
#   as.integer((thres.rng & !target) | (!thres.rng & target))
# }
# # debug(fn.target.to.xor)
# 
# fn.target.from.xor <- function(pred, data.df, col, thres) {
#   thres.rng <- data.df[[col]] > thres
#   
#   new.pred <- pred
#   new.pred[thres.rng] <-  1 - new.pred[thres.rng]
#   new.pred
# }
# 
# fn.extra.col.xor <- function(data.df, col, thres) {
#   as.factor(as.integer(data.df[[col]] > thres))
# }

##############################################################
## drop values from train not in test or val
##############################################################
fn.drop.uncommon <- function(tr, val = NULL, test) {
    for (col.name in colnames(tr)[-1]) {
      if (col.name != "ACTION") {
        col.is.factor <- is.factor(tr[[col.name]])
        if (col.is.factor) {
          tr[[col.name]] <- as.integer(as.character(tr[[col.name]]))
          if (!is.null(val)) {
            val[[col.name]] <- as.integer(as.character(val[[col.name]]))
          }
          test[[col.name]] <- as.integer(as.character(test[[col.name]]))
        }
        
        col.keep <- unique(test[[col.name]])
        if (!is.null(val)) {
          col.keep <- unique(c(col.keep, val[[col.name]]))
        }
        col.keep <- col.keep[col.keep %in% unique(tr[[col.name]])]
        
        tr[[col.name]][!(tr[[col.name]] %in% col.keep)] <- -1
        if (!is.null(val)) {
          val[[col.name]][!(val[[col.name]] %in% col.keep)] <- -1
        }
        test[[col.name]][!(test[[col.name]] %in% col.keep)] <- -1
        if (col.is.factor) {
          col.keep <- sort(c(-1,col.keep))
          tr[[col.name]] <- factor(tr[[col.name]], levels = col.keep)
          if (!is.null(val)) {
            val[[col.name]] <- factor(val[[col.name]], levels = col.keep)
          }
          test[[col.name]] <- factor(test[[col.name]], levels = col.keep)
        }
      }
  }
  list(tr = tr, val = val, test = test)
}
# debug(fn.drop.uncommon)

##############################################################
## reencode values based on their rank
##############################################################
fn.reencode <- function (tr, val = NULL, test) {
  library("data.table")
  out.tr <- tr
  out.val <- NULL
  if (!is.null(val)) {
    out.val <- val
  }
  out.test <- test
  for (col.name in colnames(tr)) {
    if (col.name != "ACTION") {
      cat("reeconding ", col.name, "...\n")
      out.tr[[col.name]] <- as.integer(as.character(out.tr[[col.name]]))
      if (!is.null(val)) {
        out.val[[col.name]] <- as.integer(as.character(out.val[[col.name]]))
      }
      out.test[[col.name]] <- as.integer(as.character(out.test[[col.name]]))
      model.lme <- suppressWarnings(fn.train.lmer(
        formula= as.formula(paste0("ACTION ~ 1 + (1|", col.name, ")")),
        data = out.tr,
        family = binomial(),
        verbose = F))
      
      col.reenc <- unique(rbind(
        out.tr[,col.name, drop = F],
        out.test[,col.name, drop = F]))
      if (!is.null(val)) {
        col.reenc <- unique(rbind(col.reenc,
                                  out.val[,col.name, drop = F]))
      }
      col.reenc[[col.name]] <- as.integer(as.character(col.reenc[[col.name]]))
      col.reenc$RANK <- model.lme$predict(col.reenc)
      col.reenc$RANK <- order(col.reenc$RANK)
#       col.reenc$RANK <- rank(col.reenc$RANK, ties.method = "min")
#       col.reenc$RANK <- factor(col.reenc$RANK)
      
      col.reenc <- data.table(col.reenc, key = col.name)
      
      out.tr[[col.name]] <- col.reenc[J(out.tr[[col.name]])]$RANK
      if (!is.null(val)) {
        out.val[[col.name]] <- col.reenc[J(out.val[[col.name]])]$RANK
      }
      out.test[[col.name]] <- col.reenc[J(out.test[[col.name]])]$RANK
    }
  }
  list(tr = out.tr, val = out.val, test = out.test)
}

##############################################################
## factor to int
##############################################################
fn.factor.2.int <- function (tr, val = NULL, test) {
  library("data.table")
  out.tr <- tr
  out.val <- val
  out.test <- test
  for (col.name in colnames(tr)) {
    if (col.name != "ACTION") {
      out.tr[[col.name]] <- as.integer(as.character(out.tr[[col.name]]))
      if (!is.null(val)) {
        out.val[[col.name]] <- as.integer(as.character(out.val[[col.name]]))
      }
      out.test[[col.name]] <- as.integer(as.character(out.test[[col.name]]))
    }
  }
  list(tr = out.tr, val = out.val, test = out.test)
}

##############################################################
## dor xor in prediction
##############################################################
fn.xor.pred <- function(pred, xor.sel) {
  xor.sel <- as.logical(xor.sel)
  pred[xor.sel] <- 1-pred[xor.sel]
  pred
}

##############################################################
## dor xor in prediction
##############################################################
fn.find.ntile <- function(val, ntile) {
  val <- sort(val)
  val[round(length(val)*ntile)]
}

#############################################################
# print rf importance
#############################################################
fn.rf.print.imp <- function(rf) {
 imp <- try(print(data.frame(importance=rf$importance[order(-rf$importance[,1]),]), 
      silent = T))
}

#############################################################
# load ensenble datset
#############################################################
fn.ens.load.df <- function(name, cols.ens) {
  fn.load.data(paste0("data.", name))
  ens.df <-  get(paste0("data.", name))[, "ACTION", drop = F]
  for (df.name in cols.ens) {
    cur.name <- paste0("data.", name, ".", df.name)
    fn.load.data(cur.name)
    ens.df[,paste0(colnames(get(cur.name)),".", df.name)] <- get(cur.name)
  }
  colnames(ens.df) <- 
    gsub(".", "_", gsub("pred.", "", 
                        tolower(colnames(ens.df)), fixed = T), 
         fixed = T) 
  ens.df
}

#############################################################
# resplit training set
#############################################################
fn.split.factors <- function(data.tr.split, 
                             data.val.split,
                             data.test.split,
                             split.list) {
  
  library("cvTools")
  library("data.table")
  data.all.split <- rbind(data.tr.split,
                          data.val.split,
                          data.test.split)
  data.all.split.type <- c(rep("tr", nrow(data.tr.split)),
                           rep("val", nrow(data.val.split)),
                           rep("test", nrow(data.test.split)))
  data.all.tr.idx <- data.all.split.type == "tr"
  for (col.name in names(split.list)) {
    lvl.values <- levels(data.all.split[[col.name]])
    rnd.split <- cvFolds(n = length(lvl.values), K = split.list[[col.name]])
    for (s.k in 1:rnd.split$K) {
      col.name.s <- paste(col.name, s.k, sep="_")
      col.keep <- lvl.values[rnd.split$subsets[rnd.split$which == s.k]]
      data.all.split[[col.name.s]] <- 
        as.integer(as.character(data.all.split[[col.name]]))
      data.all.split[[col.name.s]][
        !(data.all.split[[col.name.s]] %in% col.keep)] <- -1
      shuff.dt <- data.table(OLD = unique(data.all.split[[col.name.s]]))
      shuff.dt$NEW <- sample(nrow(shuff.dt))
      setkeyv(shuff.dt, "OLD")
      data.all.split[[col.name.s]] <- 
        shuff.dt[J(data.all.split[[col.name.s]])]$NEW
#       data.all.split.tr.only <- unique(data.all.split[[col.name.s]][
#         data.all.tr.idx])
#       data.all.split.test.only <- unique(data.all.split[[col.name.s]][
#         !data.all.tr.idx])
#       data.all.split.tr.only.lvl <- data.all.split.tr.only[
#         !(data.all.split.tr.only %in% data.all.split.test.only)]
#       data.all.split.test.only.lvl <- data.all.split.test.only[
#         !(data.all.split.test.only %in% data.all.split.tr.only)]
#       data.exclude <- c(data.all.split.tr.only.lvl,
#                         data.all.split.test.only.lvl)
#       data.all.split[[col.name.s]][
#         (data.all.split[[col.name.s]] %in% data.exclude)] <- "-2"
#       data.all.split[[col.name.s]] <- factor(data.all.split[[col.name.s]])
    }
    data.all.split[[col.name]] <- NULL
  }
  
  list(
    tr = data.all.split[data.all.split.type == "tr",],
    val = data.all.split[data.all.split.type == "val",],
    test = data.all.split[data.all.split.type == "test",])
}

assign_random_values <- function(var){
  varUnique <- unique(var)
  len <- length(varUnique)
  vals <- sample(len, len)
  df.unique <- data.frame(var = varUnique, newvar = vals)
  df <- data.frame(id=c(1:length(var)), var=var)
  df <- merge(df,df.unique,by="var",all.x=TRUE)
  df[order(df$id),"newvar"]
}

findBestCombination <- function(predicted,actual,a) {
  best_error <- 0
  predicted <- as.matrix(predicted)
  for (i in 1:nrow(a)) {
    #if (i %% 100000 == 0) print (i)
    error <- auc(actual,predicted %*% (a[i,]/sum(a[i,])))
    if (error > best_error) {
      print (error)
      best_error <- error
      best_i <- i
    }
  }
  return (a[best_i,])
}
