from scripts.utils import get_files
from collections import defaultdict
import math
import pickle

def convert_line2idx(line, vocab):

    line_data = []
    for charac in line:
        if charac not in vocab.keys():
            line_data.append(vocab["<unk>"])
        else:
            line_data.append(charac)
    return line_data

def convert_files2idx(files, vocab):

    data = []

    for file in files:
        with open(file) as f:
            lines = f.readlines()

        for line in lines:
            toks = convert_line2idx(line, vocab)
            data.append(toks)

    return data


def train(train_data, test_data):

  for line in train_data:
      chars = line
      for i in range(0,len(chars)):
          if i >= 2:
              trigrams[(chars[i-3], chars[i-2], chars[i-1])] += 1
          if i >= 3:
              fourgrams[(chars[i-3], chars[i-2], chars[i-1], chars[i])] += 1

  V = len(vocabulary)

# compute perplexity
  perplexity=0
  for idx, chars in enumerate(test_data):

      n = len(chars) #length of line
      prob_sum = 0
      denominator = 0
      numerator = 0
      loss = 0

      for i in range(3,len(chars)):
          trigram = (chars[i-3], chars[i-2], chars[i-1])
          fourgram = (chars[i-3], chars[i-2], chars[i-1], chars[i]) # fourgram from test dataset
       
          if trigram in trigrams:
            denominator = trigrams[trigram]
            
          if fourgram in fourgrams:
            numerator = fourgrams[fourgram]

          prob = (numerator+1)/(denominator+V)

          prob_sum += (math.log2(prob)) #summation of probabilities


      loss = -(prob_sum/n) #perplexity of each line
    
      perplexity +=  2**loss

  return perplexity/len(test_data)
  
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', type=str, default="/scratch/general/vast/u1448738/mp3")
    parser.add_argument('--input_dir', type=str, default="/uufs/chpc.utah.edu/common/home/u1448738/mp3/")
    parser.add_argument('--vocab_file', type=str, default="vocab.pkl")
	
	with open (f'{args.input_dir}{args.vocab_file}','rb') as f :
    	vocabulary = pickle.load (f)

	train_probabilities = defaultdict(float)
	trigrams = defaultdict(int)
	fourgrams = defaultdict(int)

	train_files = get_files(f'{args.input_dir}/data/train')
	train_data = convert_files2idx(train_files,vocabulary)

	test_files = get_files(f'{args.input_dir}/data/test')
	test_data = convert_files2idx(test_files,vocabulary)
	
	perplexity = train(train_data, test_data)
	print(perplexity)
	
	len(trigrams)

	len(fourgrams)


