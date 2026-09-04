from data_factory import tfidf
from configuration import file_space

N = 5


tfidf(file_space.train_file, file_space.test_file, file_space.test_simi_file, N)
tfidf(file_space.train_file, file_space.train_file, file_space.train_simi_file, N)
# tfidf(file_space.train_file, file_space.valid_file, file_space.valid_simi_file, N)