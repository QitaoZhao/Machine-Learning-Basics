class Book():
    """Define a class reading and analysing the text"""
    def __init__(self):
        self.num_words = 0
        # A list (['word', ...]) updates each time Book read()
        self.text = []
        # A dict ({'word': num_occur, ...}) updates each time Book countWords()
        self.stat = {}

    def read(self, route):
        """Output processed text"""
        text = open(route).read()
        # Change all characters into lowercase
        text = text.lower()
        # Replace symbols with spaces
        for i in '!"\'#$%&()*+,-./:;<=>?@[\\]^_‘{|}~':
            text = text.replace(i, " ") 
        text = text.split()
        self.num_words += len(text) 
        self.text.extend(text)

    def countWords(self):
        """Count words in the text"""
        self.stat = {}
        counts = {}
        for word in self.text:
            counts[word] = counts.get(word, 0) + 1  
        items = list(counts.items())
        # Sort by the second item in each element of the list
        items.sort(key=lambda x:x[1], reverse=True)
        for i in range(len(items)):
            word, count = items[i]
            self.stat[word] = count

    def show(self, n=10):
        """Show the vocabulary"""
        items = list(self.stat.items())
        items.sort(key=lambda x:x[1], reverse=True)
        # for i in range(len(items)):
        for i in range(n):
            word, count = items[i]
            print("{0:<10}{1:>5}    {2:>.6f}".format(word, count, (count/self.num_words)))
        print("Total number:", self.num_words)

