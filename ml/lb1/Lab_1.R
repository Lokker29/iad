#%%

require(caTools)
require(ggplot2)
require(dplyr)

#%%

# set = 'iris'
# data = iris
# target = 'Species'

# set = 'wine'
# data = read.csv('data/wine.csv')
# target = 'Wine'

# set = 'balance-scale'
# data = read.csv('data/balance-scale.csv')
# target = 'Class'
# data[target] = as.factor(sapply(data[target], as.factor))

set = 'crx'
data = read.csv('data/crx.csv', na.strings=c(""))
target = 'A16'
data[target] = as.factor(sapply(data[target], as.factor))

data

#%%

SEED = 1
set.seed(SEED)

#%%

split_data <- function(data, ratio) {
  split = sample.split(data, SplitRatio = ratio)
  
  train = subset(data, split == TRUE)
  test = subset(data, split == FALSE)
  
  return(list(train, test))
}

#%%

shuffle_data <- function(data) {
  random_indexes = sample(nrow(data))
  return(data[random_indexes,])
}

#%%

replace_target <- function(data, target) {
  data[target] = sapply(data[, target], as.numeric)
  return(data)
}

#%%

getmode <- function(v) {
  uniqv <- unique(v, na.rm = True)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

#%%

fill_empty_values <- function(data, target) {
  for (i in colnames(data[, !names(data) %in% c(target)])) {
    for (cls in levels(data[, target])) {
      if (class(data[, i]) != "character") {
        data[is.na(data[, i]) & data[target] == cls, i] = 
          mean(filter(data, data[target] == cls)[, i], na.rm = TRUE)
      } else {
        data[is.na(data[, i]) & data[target] == cls, i] = 
          getmode(filter(data, data[target] == cls)[, i])
      }
    }
  }
  return(data)
}

#%%

# Shuffle data 
data = shuffle_data(data)

# Split to train and test
splited_data = split_data(data, 0.7)
train = splited_data[[1]]
test = splited_data[[2]]

# Replace target by numeric
data = fill_empty_values(data, target)
data = replace_target(data, target)

data

#%%

if (set == 'iris') {
  plot(data$Sepal.Length, data$Sepal.Width, main='Scatterplot', pch=21, 
       bg=c("red", "green", "blue")[unclass(data$Species)])
  
  legend("topleft", legend=levels(iris$Species), col=c("red", "green", "blue"), pch=16)
  
  ggplot(iris, aes(x=Petal.Length)) + geom_histogram(aes(color=Species), bins=30) + ggtitle("Histogram")
}

#%%

if (set == 'balance-scale') {
  data_ = read.csv('data/balance-scale.csv')
  
  ggplot(data_, aes(x=Left.Distance*Left.Weight - Right.Distance*Right.Weight)) + geom_histogram(aes(color=Class), bins=40) + ggtitle("Histogram")
}

#%%

if (set == 'balance-scale') {
  data_ = read.csv('data/balance-scale.csv')
  df = data.frame(
    group = as.data.frame(table(data_$Class))[[1]],
    value = as.data.frame(table(data_$Class))[[2]]
  )
  df$perc = round(df$value/sum(df$value) * 100)
  df$pos = count(data_)[[1]] - (cumsum(df$value) - sapply(df$value, function(x) cumsum(x) - 0.5 * x))
  
  ggplot(df, aes(x="", y=value, fill=group)) + 
    geom_bar(stat="identity", width=1) + geom_text(aes(x="", y=pos, label=paste0(perc,"%"))) +
    coord_polar("y", start=0) 
}

#%%
