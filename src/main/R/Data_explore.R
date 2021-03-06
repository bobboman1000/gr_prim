
# install.packages("anytime")
# install.packages("farff")
# install.packages("xlsx")

# wd <- getwd()
# setwd("..\\..")
# parent <- getwd()
# setwd(paste0(parent, "/resources/data"))
# 
# dir.create("cleaned", showWarnings = FALSE)

setwd("C:\\Projects\\6_PRIM_RF_real\\gr_prim\\resources\\data\\")

data.summary <- function(d){
  for(i in 1:ncol(d)){
    d[, i] <- as.numeric(d[, i])
  }
  un.share <- round(apply(d, 2, function(x) length(unique(x))), 5)
  maxrep.share <- round(apply(d, 2, function(x) max(table(x[!is.na(x)])))/nrow(d), 5)
  list(un.share, maxrep.share)
}



# occupancy
library(anytime)
d <- read.table("occupancy_data\\datatest.txt", sep = ",")
d <- rbind(d, read.table("occupancy_data\\datatest2.txt", sep = ","))
d <- rbind(d, read.table("occupancy_data\\datatraining.txt", sep = ","))
d$date <- as.numeric(anytime(d$date) - anytime(anydate(d$date)) + 3600)

res <- data.summary(d)
print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))
head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d, "cleaned\\occupancy.csv", row.names = FALSE)

d <- na.omit(d)
ncol(d)
nrow(d)
table(d$Occupancy)[2]/nrow(d)


# credit cards
library("xlsx")
d <- read.xlsx2("credit_cards.xls", sheetIndex = 1, startRow = 2)
res <- data.summary(d)
print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))

head(d)
keep <- c((2:ncol(d))[!((2:ncol(d))%in%(which(res[[1]] < 20)))], ncol(d))
d <- d[, keep]
colnames(d)[ncol(d)] <- "default"
write.csv(d, "cleaned\\credit_cards.csv", row.names = FALSE)

ncol(d)
nrow(d)
table(d$default)[2]/nrow(d)


# sylva
d <- read.csv("sylva_prior.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 20)))], ncol(d))
d <- d[,keep]
write.csv(d, "cleaned\\sylva.csv", row.names = FALSE)

d <- na.omit(d)
ncol(d)
nrow(d)
table(d$label)[2]/nrow(d)


# higgs
d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
res <- data.summary(d)

head(d)
keep <- c(1, (ncol(d) - 6):ncol(d))
d <- d[, keep]
for(i in 2:ncol(d)){
  d[, i] <- as.numeric(d[, i])
}
d <- na.omit(d)
write.csv(d, "cleaned\\higgs.csv", row.names = FALSE)

ncol(d)
nrow(d)
table(d$class)[2]/nrow(d)



# higgs_o
d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
d <- d[, -((ncol(d) - 6):ncol(d))]
res <- data.summary(d)
print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))

head(d)
keep <- c(1,(1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 20)))])
d <- d[, keep]
for(i in 2:ncol(d)){
  d[, i] <- as.numeric(d[, i])
}
d <- na.omit(d)
write.csv(d, "cleaned\\higgs_o.csv", row.names = FALSE)

ncol(d)
nrow(d)
table(d$class)[2]/nrow(d)


# higgs_of
d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
d <- d[, -((ncol(d) - 6):ncol(d))]
res <- data.summary(d)

for(i in 2:ncol(d)){
  d[, i] <- as.numeric(d[, i])
}
d <- na.omit(d)
write.csv(d, "cleaned\\higgs_of.csv", row.names = FALSE)

ncol(d)
nrow(d)
table(d$class)[2]/nrow(d)


# sensorless
d <- read.table("Sensorless_drive_diagnosis.txt")
res <- data.summary(d)
print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))
write.csv(d, "cleaned\\sensorless.csv", row.names = FALSE)

d <- na.omit(d)
ncol(d)
nrow(d)
table(d$V49)[1]/nrow(d)


# jm1
d <- read.csv("jm1.csv", stringsAsFactors = FALSE)
for(i in 1:(ncol(d) - 1)){
  d[, i] <- as.numeric(d[, i])
}
d <- na.omit(d)
# res <- data.summary(d)
for(i in 1:(ncol(d) - 1)){
   d[, i] <- log(d[, i] + 1)
#   plot(density(d[, i]))
}
d$defects <- ifelse(d$defects == "true", 1, 0)
write.csv(d, "cleaned\\jm1l.csv", row.names = FALSE)
ncol(d)
nrow(d)
table(d$defects)[2]/nrow(d)


# library(FNN)
# d <- read.table("avila\\avila.txt", sep = ",")
# for(i in 1:10){
#   l <- min(knn.dist(unique(d[, i]), 1)/3)
#   d[, i] <- d[, i] + runif(nrow(d), 0, l)
# }
# res <- data.summary(d)
# d <- na.omit(d)
# ncol(d)
# nrow(d)
# table(d$V11)[1]/nrow(d)







################
################
################

d <- read.csv("numerai.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$attribute_21)[2]/nrow(d)


d <- read.table(file = 'ring.tsv', sep = '\t', header = TRUE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$target)[2]/nrow(d)


d <- read.table(file = 'shuttle.tsv', sep = '\t', header = TRUE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$target)[1]/nrow


d <- read.csv("eeg-eye-state.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$Class)[1]/nrow(d)


d <- read.csv("jm1.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$defects)[2]/nrow(d)


library("farff")
d <- readARFF("bankruptcy\\3year.arff")
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$class)[2]/nrow(d)


d <- read.csv("gammatelescope\\magic04.data", stringsAsFactors = FALSE, header = FALSE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$V11)[1]/nrow(d)


d <- read.table("avila\\avila.txt", sep = ",")
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$V11)[1]/nrow(d)


d <- read.csv("HTRU_2.csv", stringsAsFactors = FALSE, header = FALSE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$V9)[2]/nrow(d)


d <- read.table(file = 'clean2.tsv', sep = '\t', header = TRUE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$target)[2]/nrow(d)


d <- read.csv("gas_sensors.csv", stringsAsFactors = FALSE, header = TRUE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$y)[1]/nrow(d)


d <- read.csv("seizure.csv", stringsAsFactors = FALSE, header = TRUE)
res <- data.summary(d)
d <- na.omit(d)
ncol(d)
nrow(d)
table(d$y)[1]/nrow(d)






# 
# 
# # # no
# # d <- read.csv("electricity-normalized.csv", stringsAsFactors = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# 
# # ok. 8 col
# d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# head(d)
# # keep <- c(1,(1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
# keep <- c(1, (ncol(d) - 6):ncol(d))
# d <- d[, keep]
# for(i in 2:ncol(d)){
#   d[, i] <- as.numeric(d[, i])
# }
# d <- na.omit(d)
# write.csv(d, "cleaned\\higgs.csv", row.names = FALSE)
# ncol(d)
# colnames(d)
# 
# 
# # ok. 18 col
# 
# d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
# d <- d[, -((ncol(d) - 6):ncol(d))]
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 10), "/", length(res[[1]])))
# 
# head(d)
# keep <- c(1,(1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 10)))])
# d <- d[, keep]
# for(i in 2:ncol(d)){
#   d[, i] <- as.numeric(d[, i])
# }
# d <- na.omit(d)
# write.csv(d, "cleaned\\higgs_o.csv", row.names = FALSE)
# ncol(d)
# colnames(d)
# 
# 
# 
# 
# # 
# # # ok. keep all! 22 col    
# # d <- read.csv("numerai.csv", stringsAsFactors = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# # ncol(d)
# # 
# # 
# # # ok. keep all! 21 col
# # d <- read.table(file = 'ring.tsv', sep = '\t', header = TRUE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# # ncol(d)
# 
# 
# # ok? CHANGE TARGET to 4! 6 col
# d <- read.table(file = 'shuttle.tsv', sep = '\t', header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# head(d)
# write.csv(d[, keep], "cleaned\\shuttle.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# # ok. CHANGE TARGET NAME! 16 col
# library("xlsx")
# d <- read.xlsx2("credit_cards.xls", sheetIndex = 1, startRow = 2)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((2:ncol(d))[!((2:ncol(d))%in%(which(res[[1]] < 20)))], ncol(d))
# d <- d[, keep]
# colnames(d)[ncol(d)] <- "default"
# write.csv(d, "cleaned\\credit_cards.csv", row.names = FALSE)
# ncol(d)
# 
# 
# # # ok. keep all! 15
# # d <- read.csv("eeg-eye-state.csv", stringsAsFactors = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# # ncol(d)
# 
# 
# 
# # check
# d <- read.table(file = 'sleep.tsv', sep = '\t', header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# head(d)
# write.csv(d[, keep], "cleaned\\shuttle.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# 
# 
# 
# # ok. do not filter additional fields. 17 col
# d <- read.csv("jm1.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# d <- d[, keep]
# colnames(d)[2] <- "v_g"
# # d <- d[, !(colnames(d) %in% c("lOCode", "lOComment", "lOBlank", "lOCode", "locCodeAndComment"))]
# write.csv(d, "cleaned\\jm1.csv", row.names = FALSE)
# ncol(d)
# 
# 
# # ok. 63 col - 
# library("farff")
# d <- readARFF("bankruptcy\\3year.arff")
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 20), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# write.csv(d[, keep], "cleaned\\bankruptcy.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# # # ok. keep all! 11 col
# # d <- read.csv("gammatelescope\\magic04.data", stringsAsFactors = FALSE, header = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# # ncol(d)
# 
# 
# #==================
# 
# 
# # ok. 21 col
# d <- read.csv("sylva_prior.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 3), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# write.csv(d[, keep], "cleaned\\sylva.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# # ok. 9 col
# d <- read.csv("click_prediction.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# head(d)
# keep <- c(1, (1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
# write.csv(d[, keep], "cleaned\\click.csv", row.names = FALSE)
# ncol(d[, keep])
# colnames(d[, keep])
# 
# 
# # ok. DO NOT CHANGE TARGET TO F. 10 col
# d <- read.table("avila\\avila.txt", sep = ",")
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# write.csv(d[, keep], "cleaned\\avila.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# # ok. 21 col
# d <- read.csv("SAAC2.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# for(i in 1:ncol(d)){
#   d[, i] <- as.numeric(d[, i])
# }
# 
# head(d)
# keep <- c(1, (1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
# write.csv(d[, keep], "cleaned\\SAAC2.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# # ok. 4 col
# d <- read.csv("mozilla4.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# write.csv(d[, keep], "cleaned\\mozilla.csv", row.names = FALSE)
# ncol(d[, keep])
# 
# 
# 
# 
# # # ok. keep all! 9 cols
# # d <- read.csv("HTRU_2.csv", stringsAsFactors = FALSE, header = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# 
# # ok. 162 col
# d <- read.table(file = 'clean2.tsv', sep = '\t', header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# head(d)
# keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
# d <- d[, keep]
# d <- d[, -1] # this one is too correlated with the target. so make it more complex!
# write.csv(d, "cleaned\\clean2.csv", row.names = FALSE)
# ncol(d)
# 
# 
# # # ok. keep all! 129
# # d <- read.csv("gas_sensors.csv", stringsAsFactors = FALSE, header = TRUE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# # 
# # 
# # # ok. delete the first column. 179 cols
# # d <- read.csv("seizure.csv", stringsAsFactors = FALSE, header = TRUE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# 
# 
# # ===
# 
# 
# # ok. 49 cols
# d <- read.table("Sensorless_drive_diagnosis.txt")
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# write.csv(d, "cleaned\\sensorless.csv", row.names = FALSE)
# ncol(d)
# 
# 
# 
# 
# # ===========
# 
# 
# # d <- read.csv("nomao.csv", stringsAsFactors = FALSE)
# # res <- data.summary(d)
# # print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# 
# 
# 
# ## try sleep data from Benjamin's code