
# install.packages("anytime")
# install.packages("farff")
# install.packages("xlsx")

wd <- getwd()
setwd("..\\..")
parent <- getwd()
setwd(paste0(parent, "/resources/data"))

dir.create("cleaned", showWarnings = FALSE)

setwd("C:\\Projects\\6_PRIM_RF_real\\gr_prim\\resources\\data\\")

data.summary <- function(d){
  for(i in 1:ncol(d)){
    d[, i] <- as.numeric(d[, i])
  }
  un.share <- round(apply(d, 2, function(x) length(unique(x))), 5)
  maxrep.share <- round(apply(d, 2, function(x) max(table(x[!is.na(x)])))/nrow(d), 5)
  list(un.share, maxrep.share)
}

# # no
# d <- read.csv("electricity-normalized.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))


# ok. 25 col
d <- read.csv("higgs.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c(1,(1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
write.csv(d[, keep], "cleaned\\higgs.csv", row.names = FALSE)
ncol(d[, keep])
colnames(d[, keep])

# 
# # ok. keep all! 22 col    
# d <- read.csv("numerai.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# ncol(d)
# 
# 
# # ok. keep all! 21 col
# d <- read.table(file = 'ring.tsv', sep = '\t', header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# ncol(d)


# ok? CHANGE TARGET to 4! 6 col
d <- read.table(file = 'shuttle.tsv', sep = '\t', header = TRUE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
head(d)
write.csv(d[, keep], "cleaned\\shuttle.csv", row.names = FALSE)
ncol(d[, keep])


# ok. CHANGE TARGET NAME! 16 col
library("xlsx")
d <- read.xlsx2("credit_cards.xls", sheetIndex = 1, startRow = 2)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
d <- d[, keep]
colnames(d)[ncol(d)] <- "default"
write.csv(d, "cleaned\\credit_cards.csv", row.names = FALSE)
ncol(d[, keep])


# # ok. keep all! 15
# d <- read.csv("eeg-eye-state.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# ncol(d)


# ok. do not filter additional fields. 17 col
d <- read.csv("jm1.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
d <- d[, keep]
colnames(d)[2] <- "v_g"
# d <- d[, !(colnames(d) %in% c("lOCode", "lOComment", "lOBlank", "lOCode", "locCodeAndComment"))]
write.csv(d, "cleaned\\jm1.csv", row.names = FALSE)
ncol(d)


# ok. 63 col
library("farff")
d <- readARFF("bankruptcy\\3year.arff")
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d[, keep], "cleaned\\bankruptcy.csv", row.names = FALSE)
ncol(d[, keep])


# # ok. keep all! 11 col
# d <- read.csv("gammatelescope\\magic04.data", stringsAsFactors = FALSE, header = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# ncol(d)


#==================


# ok. 21 col
d <- read.csv("sylva_prior.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d[, keep], "cleaned\\sylva.csv", row.names = FALSE)
ncol(d[, keep])


# ok. 9 col
d <- read.csv("click_prediction.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c(1, (1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
write.csv(d[, keep], "cleaned\\click.csv", row.names = FALSE)
ncol(d[, keep])
colnames(d[, keep])


# ok. DO NOT CHANGE TARGET TO F. 10 col
d <- read.table("avila\\avila.txt", sep = ",")
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d[, keep], "cleaned\\avila.csv", row.names = FALSE)
ncol(d[, keep])


# ok. 21 col
d <- read.csv("SAAC2.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

for(i in 1:ncol(d)){
  d[, i] <- as.numeric(d[, i])
}

head(d)
keep <- c(1, (1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))])
write.csv(d[, keep], "cleaned\\SAAC2.csv", row.names = FALSE)
ncol(d[, keep])


# ok. 4 col
d <- read.csv("mozilla4.csv", stringsAsFactors = FALSE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d[, keep], "cleaned\\mozilla.csv", row.names = FALSE)
ncol(d[, keep])


# ok. change date to number. 6 col - don't use
library(anytime)
d <- read.table("occupancy_data\\datatest.txt", sep = ",")
d <- rbind(d, read.table("occupancy_data\\datatest2.txt", sep = ","))
d <- rbind(d, read.table("occupancy_data\\datatraining.txt", sep = ","))
d$date <- as.numeric(anytime(d$date) - anytime(anydate(d$date)) + 3600)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
write.csv(d[, keep], "cleaned\\occupancy.csv", row.names = FALSE)
ncol(d[, keep])
write.csv(d, "cleaned\\occupancy_all.csv", row.names = FALSE)

# # ok. keep all! 9 cols
# d <- read.csv("HTRU_2.csv", stringsAsFactors = FALSE, header = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))


# ok. 162 col
d <- read.table(file = 'clean2.tsv', sep = '\t', header = TRUE)
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))

head(d)
keep <- c((1:ncol(d))[!((1:ncol(d))%in%(which(res[[1]] < 50 | res[[2]] > 0.25)))], ncol(d))
d <- d[, keep]
d <- d[, -1] # this one is too correlated with the target. so make it more complex!
write.csv(d, "cleaned\\clean2.csv", row.names = FALSE)
ncol(d)


# # ok. keep all! 129
# d <- read.csv("gas_sensors.csv", stringsAsFactors = FALSE, header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
# 
# 
# # ok. delete the first column. 179 cols
# d <- read.csv("seizure.csv", stringsAsFactors = FALSE, header = TRUE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))



# ===


# ok. 49 cols
d <- read.table("Sensorless_drive_diagnosis.txt")
res <- data.summary(d)
print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))
write.csv(d, "cleaned\\sensorless.csv", row.names = FALSE)
ncol(d)




# ===========


# d <- read.csv("nomao.csv", stringsAsFactors = FALSE)
# res <- data.summary(d)
# print(paste0(which(res[[1]] < 50 | res[[2]] > 0.25), "/", length(res[[1]])))




## try sleep data from Benjamin's code