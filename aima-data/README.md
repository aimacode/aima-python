# `aima-data`
Data files to accompany the algorithms from Norvig And Russell's *Artificial Intelligence - A Modern Approach*.
The idea is that the same data files can be used with any of the implementations: `aima-java`, `aima-python`, `aima-lisp`, or eventually `aima-javascript`.

The data is divided into three types:

1. Range data for Robot Localization
2. Machine Learning data sets
3. English text

We'll list the source files for each of the three:

## Range data for Robot Localization
```text
    ascii-robotdata1.log        The data
    ascii-robotdata1.txt        Description of the data
```

## Machine Learning Data Sets
```text
    iris.csv                        Data on different types of iris flowers
    orings.csv                      Data from O-rings on space shuttle missions
    restaurant.csv                  Restaurant example from the textbook
    zoo.csv                         Animals and their characteristics

    iris.txt                        Descriptions of the above files
    orings.txt
    zoo.txt

    MNIST/
        train-images-idx3-ubyte     60,000 training images
        train-labels-idx1-ubyte     60,000 labels for training images
        t10k-images-idx3-ubyte      10,000 testing images
        t10k-labels-idx1-ubyte      10,000 labels for testing images
        mnist.txt                   Description of MNIST files
```

## English text
```text
    MAN/
        *.txt               Various Unix "man page" entries
    EN-text/
        wordlist.txt        A list of 173000 English words
        spam.txt            Collection of spam emails
        gutenberg.txt       Copyright notice from Project Gutenberg, where rest comes from
        flatland.txt        The novel Flatland
        pride.txt           Pride and Predjudice
        sense.txt           Sense and Sensibility
        zola.txt            Works of Emile Zola
        sgb-words.txt       The list of 5757 five-letter English words
```
