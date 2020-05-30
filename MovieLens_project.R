# Section 1: Create edx set, validation set
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")
# if using R 3.5 or earlier, use `set.seed(1)` instead
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


# Section 2: Using edx set to train the model
# Create the train set and test set from edx
set.seed(1,  sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1, p = 0.2, list = FALSE)
test_set <- edx[test_index,]
train_set <- edx[-test_index,]

# Data visualization
edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies")

edx %>%
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() +
  ggtitle("Users")


# Define the RMSE function
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

# Just the average
mu_hat <- mean(train_set$rating)
naive_rmse <- RMSE(test_set$rating, mu_hat)
rmse_results <- tibble(method = "Just the average", RMSE = naive_rmse)
rmse_results %>% knitr::kable()

# Model 1: Movie Effect Model
mu <- mean(train_set$rating) 
movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu))

predicted_set <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>% 
  mutate(pred_rating = mu + b_i) 

# Find out the predicted_set$pred_rating = NA
predicted_set %>% 
  filter(is.na(pred_rating)) %>% 
  nrow()

# Filling the NA with a value which is lower than average mu
# Refer to Machine Learning course material - Section 6.2 (Comprehension Check Question 4)
fill_na <- round(mu - 0.5)
predicted_set$pred_rating[is.na(predicted_set$pred_rating)] <- fill_na

# Model 1: RMSE
model_1_rmse <- RMSE(test_set$rating, predicted_set$pred_rating)
rmse_results_1 <- bind_rows(rmse_results,
                          tibble(method="Movie Effect Model",
                                     RMSE = model_1_rmse ))
rmse_results_1 %>% knitr::kable()


# Model 2: Movie + User Effects Model
user_avgs <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))

predicted_set_2 <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred_rating = mu + b_i + b_u) 

# Filling the NA with a value which is lower than average mu
# Refer to Machine Learning course material - Section 6.2 (Comprehension Check Question 4)
fill_na <- round(mu - 0.5)
predicted_set_2$pred_rating[is.na(predicted_set_2$pred_rating)] <- fill_na

# Model 2: RMSE
model_2_rmse <- RMSE(test_set$rating, predicted_set_2$pred_rating)
rmse_results_2 <- bind_rows(rmse_results_1,
                          tibble(method="Movie + User Effects Model",  
                                     RMSE = model_2_rmse ))
rmse_results_2 %>% knitr::kable()


# Model 3: Regularized Movie Effect Model
# To determine the tuning parameter lambdas
lambdas <- seq(0, 10, 0.25)

mu <- mean(train_set$rating)

model_3_rmse <- sapply(lambdas, function(l){
  movie_reg_avgs <- train_set %>% 
    group_by(movieId) %>% 
    summarize(b_i = sum(rating - mu)/(n()+l), n_i = n()) 
  predicted_ratings <- test_set %>% 
    left_join(movie_reg_avgs, by='movieId') %>% 
    mutate(pred = mu + b_i) 
  fill_na <- round(mu - 0.5)
  predicted_ratings$pred[is.na(predicted_ratings$pred)] <- fill_na
  return(RMSE(predicted_ratings$pred, test_set$rating))
})
qplot(lambdas, model_3_rmse)  
lambdas[which.min(model_3_rmse)]
min(model_3_rmse)

# Model 3: RMSE
rmse_results_3 <- bind_rows(rmse_results_2,
                          tibble(method="Regularized Movie Effect Model",  
                                     RMSE = min(model_3_rmse)))
rmse_results_3 %>% knitr::kable()


# Model 4: Regularized Movie + User Effect Model
lambdas <- seq(0, 10, 0.25)
model_4_rmse <- sapply(lambdas, function(l){
  mu <- mean(train_set$rating)
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  predicted_ratings <- test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    mutate(pred = mu + b_i + b_u) 
  fill_na <- round(mu - 0.5)
  predicted_ratings$pred[is.na(predicted_ratings$pred)] <- fill_na
  return(RMSE(predicted_ratings$pred, test_set$rating))
})

qplot(lambdas, model_4_rmse)  

lambdas[which.min(model_4_rmse)]
min(model_4_rmse)

# Model 4: RMSE
rmse_results_4 <- bind_rows(rmse_results_3,
                          tibble(method="Regularized Movie + User Effect Model",  
                                     RMSE = min(model_4_rmse)))
rmse_results_4 %>% knitr::kable()
