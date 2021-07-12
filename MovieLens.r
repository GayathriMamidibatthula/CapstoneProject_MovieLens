## Loading the dataset

#############################################################
# Create edx set, validation set, and submission file
#############################################################


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]
# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)
###################################################################################
#################################################

##Loading libraries

library(tidyverse)
library(ggplot2)
library(dplyr)
library(markdown)
library(knitr)
library(caret)
library(data.table)
library(lubridate)

#Checking for any null values in the dataset:
anyNA(edx)
##Exploring data - Exploratory Data Analysis
head(edx)
head(validation)

## No of rows and columns in the dataset: 

rows<- dim(edx)[1]
columns<- dim(edx)[2]
summary(rows)
summary(columns)

## No of unique movies and users:

uniqueMovies <- length(unique(edx$movieId))
uniqueUsers<- length(unique(edx$userId))

## Movie Rating based on genre:
genreList <- c('Drama', 'Comedy', 'Thriller', 'Romance')
genreCounts <- sapply(genreList, function(g){
  edx %>% filter(str_detect(genres, g)) %>% tally()
})

edx %>% separate_rows(genres, sep = "\\|") %>%
  group_numRatings <- edx %>% group_by(movieId) %>% 
  summarize(numRatings = n(), movieTitle = first(title)) %>%
  arrange(desc(numRatings)) %>%
  top_n(10, numRatings)by(genres) %>%
  summarize(count = n())

## Movie with greatest number of ratings:

numRatings <- edx %>% group_by(movieId) %>% 
  summarize(numRatings = n(), movieTitle = first(title)) %>%
  arrange(desc(numRatings)) %>%
  top_n(10, numRatings)

##Histogram representation of no of ratings based on movies:
edx %>% count(movieId) %>% ggplot(aes(n))+
  geom_histogram(color = "black" , fill= "white")+
  scale_x_log10()+
  ggtitle("Rating Count Per Movie")+
  theme_gray()

#Histogram of number of reviews per movie id
edx %>%
  group_by(movieId) %>%
  summarise(n_reviews=n()) %>%
  ggplot(aes(n_reviews)) +
  geom_histogram(color="white") +
  scale_x_log10() +
  ggtitle("Histogram of number of reviews for each movie")

## DATA WRANGLING

##splitting edx sample to train and test set with ratio 80:20

train_index <- createDataPartition(y=edx$rating, times=1, p=0.8,list=FALSE)

train <- edx[train_index,]
test <- edx[-train_index,]

## Making sure we only include the users and movies in test set, which are also in training set.
## Extra entries are removed from test set using semi-join function

test <- test %>% 
  semi_join(train, by = "movieId") %>%
  semi_join(train, by = "userId")

##Data Wrangling

class(train$timestamp)
##Timestamp returns an integer,hence changing timestamp to datetime, and using only the year
train$timestamp <- year(as_datetime(train$timestamp))

#extracting release year from title
pattern <- "(?<=\\()\\d{4}(?=\\))"
train$release_year <- train$title %>% str_extract(pattern) %>% as.integer()


#one-hot encode the genres column
train$genres <- str_split(train$genres, pattern="\\|")
one_hot_genres <- enframe(train$genres) %>%
  unnest(value) %>%
  mutate(temp = 1) %>%
  pivot_wider(names_from = value, values_from = temp, values_fill = list(temp = 0))
train <- cbind(train, one_hot_genres) %>% select(-name)
train$genres <- NULL

#adding the average rating  for each movie minus the total average rating
avg_rating <- mean(train$rating)
movie_score <- train %>% group_by(movieId) %>%
  summarise(movie_score = mean(rating-avg_rating))

#adding the average rating for each user minus the total average rating and movie score
user_score <- train %>% left_join(movie_score, by="movieId") %>%
  mutate(movie_score = ifelse(is.na(movie_score), 0, movie_score)) %>%
  group_by(userId) %>%
  summarise(user_score = mean(rating-avg_rating-movie_score)) 

train <- train %>% left_join(user_score) %>% left_join(movie_score)

head(train)

##Same wrangling process applied to test set:
class(test$timestamp) ##returns an integer

#changing timestamp to datetime, and using only the year
test$timestamp <- year(as_datetime(test$timestamp))

#extract release year from title
pattern <- "(?<=\\()\\d{4}(?=\\))"
test$release_year <- test$title %>% str_extract(pattern) %>% as.integer()

#one-hot encode the genres column
test$genres <- str_split(test$genres, pattern="\\|")
one_hot_genres <- enframe(test$genres) %>%
  unnest(value) %>%
  mutate(temp = 1) %>%
  pivot_wider(names_from = value, values_from = temp, values_fill = list(temp = 0))
test <- cbind(test, one_hot_genres) %>% select(-name)
train$genres <- NULL

#adding columns of genres that are not present in test set, and removing those that are not in the train set
for(col in names(train)){
  if(!col %in% names(test)){
    test$newcol <- 0
    names(test)[names(test)=="newcol"] <- col
  }
}
for(col in names(test)){
  if(!col %in% names(train)){
    test[,col] <- NULL
  }
}

#adding the average scores on the train set of each user and movie
test$user_score <- NULL
test$movie_score <- NULL
test <- test %>% left_join(user_score, by="userId") %>% left_join(movie_score, by="movieId")

#if there are users or movies in the test set that are not in the train set, assigning the score of the user and movie as 0, to prevent inconsistency
test <- test %>% mutate(user_score = ifelse(is.na(user_score), 0, user_score)) %>% mutate(movie_score = ifelse(is.na(movie_score), 0, movie_score))

#reorder the columns to follow the train set
test <- test %>% select(names(train))
head(test)

## BUILDING ML MODELS

##Calculating RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2, na.rm = TRUE))
}

##Naive RMSE - predict ratings as the mean of training set ratings

##calculate mean of the training set
train_rating_1<- mean(train$rating)
train_rating_1

##predict RMSE for Naive model
naive_rmse <- RMSE( train_rating_1,test$rating)
naive_rmse
##Predict RMSE for Linear Model:


#Linear Model with timestamp, release_year, user_score, and movie_score
control <- trainControl(method = "none")
fit_linear <- train(rating~user_score+movie_score+timestamp+release_year, data=train, method="lm", trControl=control)
print(fit_linear$finalModel)

y_hat <- predict(fit_linear, test)
linear_model <- RMSE(test$rating, y_hat)
cat("RMSE :", linear_model)

#Linear Model with regularisation, with only user_score and movie_score

#splitting the train set into 2 to calculate the best lambda
idx <- createDataPartition(train$rating, times=1, p=0.8, list=FALSE)
train_part_1 <- train[idx, ]
train_part_2 <- train[-idx, ]

#calculating the best lambda
lambdas <- seq(0, 10, 0.25)
rmses <- sapply(lambdas, function(l){
  avg_rating <- mean(train_part_1$rating)
  movie_score <- train_part_1 %>%
    group_by(movieId) %>%
    summarize(b_m = sum(rating - avg_rating)/(n()+l))
  user_score <- train_part_1 %>% 
    left_join(movie_score, by="movieId") %>%
    mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_m - avg_rating)/(n()+l))
  predicted_ratings <- 
    train_part_2 %>% 
    left_join(movie_score, by = "movieId") %>%
    left_join(user_score, by = "userId") %>%
    mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
    mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
    mutate(pred = avg_rating + b_m + b_u) %>%
    .$pred
  return(RMSE(predicted_ratings, train_part_2$rating))
})

lambda <- lambdas[which.min(rmses)]
qplot(lambdas, rmses)

print(lambda)

##Training the final model

##Lambda = 4.75. It is used to train the model and predict the test set

#prediction on test set
lambda <- 4.75
avg_rating <- mean(train$rating)
movie_score <- train %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - avg_rating)/(n()+lambda))
user_score <- train %>% 
  left_join(movie_score, by="movieId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - avg_rating)/(n()+lambda))
predicted_ratings <- 
  test %>% 
  left_join(movie_score, by = "movieId") %>%
  left_join(user_score, by = "userId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
  mutate(pred = avg_rating + b_m + b_u) %>%
  .$pred

linear_reg <- RMSE(test$rating, predicted_ratings)
cat("RMSE :", linear_reg)

##Using genres column to help with predictions
not_genres <- c("userId", "movieId", "rating", "timestamp", "title", "release_year", "user_score", "movie_score")
genres <- colnames(train)[!colnames(train) %in% not_genres]
genres

#calculating the average ratings for each genre
genre_scores <- data.frame(genre="",m=0, sd=0)
for(genre in genres){
  results <- train %>% filter(train[colnames(train)==genre]==1) %>%
    summarise(m=mean(rating), sd=sd(rating))
  genre_scores <- genre_scores %>% add_row(genre=genre, m=results$m, sd=results$sd)
}
genre_scores <- genre_scores[-1,]
genre_scores[is.na(genre_scores)] <- 0
genre_scores

#Linear regularised model with genre feature
lambda <- 4.75
avg_rating <- mean(train$rating)
movie_score <- train %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - avg_rating)/(n()+lambda))
user_score <- train %>% 
  left_join(movie_score, by="movieId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - avg_rating)/(n()+lambda))
genre_score <- as.matrix(test[, genres]) %*% genre_scores$m
n_genres <- rowSums(test[,genres])
genre_score <- genre_score / n_genres

#using the genre_scores if the user and movie is unknown

predicted_ratings <- 
  test %>% 
  left_join(movie_score, by = "movieId") %>%
  left_join(user_score, by = "userId") %>%
  cbind(genre_score) %>%
  mutate(pred = genre_score) %>%
  mutate(pred = ifelse(!is.na(b_m)|!is.na(b_u), 
                       avg_rating + replace_na(b_m,0) + replace_na(b_u,0), 
                       pred))

linear_reg_2 <- RMSE(test$rating, predicted_ratings$pred)
cat("RMSE :", linear_reg_2)

#Table of RMSE Results

data.frame(
  method=c("Naive Prediction", "Linear Model", "Linear Model with Regularisation(only using movie and user scores)", "Linear Model with Regularisation(movie, user, and genre scores)"), 
  rmse=c(naive_rmse, linear_model, linear_reg, linear_reg_2))

#Train the final model
lambda <- 4.75
avg_rating <- mean(edx$rating)
movie_score <- edx %>%
  group_by(movieId) %>%
  summarize(b_m = sum(rating - avg_rating)/(n()+lambda))
user_score <- edx %>% 
  left_join(movie_score, by="movieId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_m - avg_rating)/(n()+lambda))
predicted_ratings <- 
  validation %>% 
  left_join(movie_score, by = "movieId") %>%
  left_join(user_score, by = "userId") %>%
  mutate(b_m = ifelse(is.na(b_m), 0, b_m)) %>%
  mutate(b_u = ifelse(is.na(b_u), 0, b_u)) %>%
  mutate(pred = avg_rating + b_m + b_u) %>%
  .$pred

final_result <- RMSE(validation$rating, predicted_ratings)
cat("RMSE :", final_result)
