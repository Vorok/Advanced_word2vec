import os
from tqdm import tqdm
if __name__ == '__main__':
	FILENAME_IN = "./data/enwik8_shorter_cleaned.txt"
	FILENAME_OUT = "./data/enwik8_shorter_cleaned_for_char_embed.txt"
	with open(FILENAME_OUT, 'w') as out:
		for line in tqdm(open(FILENAME_IN, encoding="utf8")):
			line = line.split()
			for word in line:
				out.write("%s\n" % word)