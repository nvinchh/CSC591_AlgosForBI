from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark.streaming.kafka import KafkaUtils
import operator
import numpy as np
import matplotlib.pyplot as plt


def main():
    conf = SparkConf().setMaster("local[2]").setAppName("Streamer")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 10)   # Create a streaming context with batch interval of 10 sec
    ssc.checkpoint("checkpoint")

    pwords = load_wordlist("positive.txt")
    nwords = load_wordlist("negative.txt")
   
    counts = stream(ssc, pwords, nwords, 100)
    make_plot(counts)


def make_plot(counts):
    """
    Plot the counts for the positive and negative words for each timestep.
    Use plt.show() so that the plot will popup.
    """
    # YOUR CODE HERE
    pos = []    #positive counts for each time step
    neg = []    #negative counts for each time step
    for element in counts:
        dictionary = element
        for key in dictionary:
            if key[0]=='negative':
                neg.append(key[1])
            elif key[0]=='positive':
                pos.append(key[1])
    maximumValues = max(max(neg), max(pos))     #To determine the upper limit of the axes
    plt.plot(pos, 'bo-')
    plt.plot(neg, 'go-')
    plt.xlabel('Time Step')
    plt.ylabel('Word Count')
    plt.legend(['positive', 'negative'], loc='upper left')
    plt.axis([0, 12, 0, maximumValues+100])
    plt.show()

def load_wordlist(filename):
    """ 
    This function should return a list or set of words from the given filename.
    """
    # YOUR CODE HERE
    fname = open(filename)
    text = fname.read()
    listofwords = text.split('\n')
    return listofwords

def stream(ssc, pwords, nwords, duration):
    kstream = KafkaUtils.createDirectStream(
        ssc, topics = ['twitterstream'], kafkaParams = {"metadata.broker.list": 'localhost:9092'})
    tweets = kstream.map(lambda x: x[1].encode("ascii","ignore"))

    # Each element of tweets will be the text of a tweet.
    # You need to find the count of all the positive and negative words in these tweets.
    # Keep track of a running total counts and print this at every time step (use the pprint function).
    # YOUR CODE HERE
    tweets2 = tweets.flatMap(lambda x: x.split(" "))
    pairs = tweets2.map(lambda word: ("positive", 1) if word in pwords else ("negative", 1) if word in nwords else ("negative", 0))
    wordcounts = pairs.reduceByKey(lambda x, y: x+y)

    def updateFunction(newValues, last_sum):
        return sum(newValues) + (last_sum or 0)

    runningCounts = wordcounts.updateStateByKey(updateFunction)
    runningCounts.pprint()


    # Let the counts variable hold the word counts for all time steps
    # You will need to use the foreachRDD function.
    # For our implementation, counts looked like:
    #   [[("positive", 100), ("negative", 50)], [("positive", 80), ("negative", 60)], ...]
    counts = []
    wordcounts.foreachRDD(lambda t,rdd: counts.append(rdd.collect()))
    
    ssc.start()                         # Start the computation
    ssc.awaitTerminationOrTimeout(duration)
    ssc.stop(stopGraceFully=True)

    return counts

if __name__=="__main__":
    main()
