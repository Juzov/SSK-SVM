import os
#import ssk_prune as ssk
from ssk_cache import StringSubsequenceKernel
from ssk_prune import StringSubsequenceKernelWithPrune
import time
import string_functions
import svm_approx

timestr = time.strftime("%Y%m%d-%H%M%S")
path = os.path.dirname(os.path.abspath(__file__))

def get_ssk(to_prune, k,lambda_decay, multiply_factor):
    if(to_prune):
	    theta = multiply_factor*k
	    return StringSubsequenceKernelWithPrune(k,lambda_decay,theta)
    else:
	    return StringSubsequenceKernel(k,lambda_decay)

#spam or reuters? 
is_spam = True

k = 5
lambda_decay = 0.5

##TRUE: ssk_prune FALSE: ssk_cache
to_prune = True

#multiply factor for theta/m multiply_factor * k = theta
multiply_factor = 3
#Documents NOTE: EVEN NUMBERS ONLY

amount_of_documents_list = [[2,2],[2,3],[2,4]] 
#test_train_ratio = 0.3
#size of most_used
word_amount = 200


filename_lambda = timestr + '-lambda-ptest.txt'
filename_cache = timestr + '-cache-ptest.txt'
filename = filename_lambda
data_file_lambda = open(path + '/tests/' + filename_lambda, 'w+')
data_file_cache = open(path + '/tests/' + filename_cache, 'w+')


for amount_of_documents in amount_of_documents_list:
    for x in range(0, 2):
        ssk = get_ssk(to_prune, k, lambda_decay, multiply_factor)
        accuracy, elapsed_time = svm_approx.svm_calc(is_spam, amount_of_documents[0], amount_of_documents[1], ssk, word_amount, k)
        print ("Amount of Documents: ", amount_of_documents, ", To prune:", to_prune, ", Acc: ", accuracy, ", Elasped time: ", elapsed_time)
        write_string = str(amount_of_documents) + ' ' + str(accuracy) + ' ' + str(elapsed_time) + '\n'
        
        to_prune = not to_prune
        if(filename == filename_lambda):
            filename = filename_cache
            data_file_lambda.write(write_string)
        else:
            filename = filename_lambda
            data_file_cache.write(write_string)

data_file_lambda.close()
data_file_cache.close()