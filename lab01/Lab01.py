class Lab01:

    def __init__(self):
        self.next_word_counts = {}

    def fit(self, words):

        for i in range(len(words)-1):
            current_word = words[i]
            next_word = words[i+1]

            if not self.next_word_counts.get(current_word):
                self.next_word_counts[current_word] = {}


            self.next_word_counts[current_word][next_word] = \
                self.next_word_counts[current_word].get(next_word, 0) + 1    

        for i, (word, next_words) in enumerate(self.next_word_counts.items()):
            if i < 10:
               print(word, next_words) 
            # if i>= 2000:
            #     break
        

    def prob(self, v, w):
        if not self.next_word_counts.get(v) or not self.next_word_counts.get(w):
            return 0
        if sum(self.next_word_counts[v].values()) == 0:
            return 0
        return self.next_word_counts[v][w]/sum(self.next_word_counts[v].values())
