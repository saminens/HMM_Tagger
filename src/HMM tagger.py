from IPython.core.display import HTML
from collections import Counter
from src.helpers import Dataset
from pomegranate import State, HiddenMarkovModel, DiscreteDistribution

from collections import namedtuple, defaultdict

FakeState = namedtuple("FakeState", "name")
# Example from the Brown corpus.
# ```
# b100-38532
# Perhaps	ADV
# it	PRON
# was	VERB
# right	ADJ
# ;	.
# ;	.
#
# b100-35577
# ...
# ```
class MFCTagger:
    # NOTE: You should not need to modify this class or any of its methods
    missing = FakeState(name="<MISSING>")

    def __init__(self, table):
        self.table = defaultdict(lambda: MFCTagger.missing)
        self.table.update({word: FakeState(name=tag) for word, tag in table.items()})

    def viterbi(self, seq):
        """This method simplifies predictions by matching the Pomegranate viterbi() interface"""
        return 0., list(enumerate(["<start>"] + [self.table[w] for w in seq] + ["<end>"]))


# Sequence starting counts
def starting_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the beginning of
    a sequence.

    For example, if 8093 sequences start with NOUN, then you should return a
    dictionary such that your_starting_counts[NOUN] == 8093
    """
    dictionary = {}
    for seq in sequences:
        if seq[0] not in dictionary:
            dictionary[seq[0]] = 0
        else:
            dictionary[seq[0]] += 1

    return dictionary

def pair_count(sequences_A, sequences_B):
    for i in zip(*data.training_set.stream()):
        print(i)
        break

def pair_counts(sequences_A, sequences_B):
    """Return a dictionary keyed to each unique value in the first sequence list
    that counts the number of occurrences of the corresponding value from the
    second sequences list.

    For example, if sequences_A is tags and sequences_B is the corresponding
    words, then if 1244 sequences contain the word "time" tagged as a NOUN, then
    you should return a dictionary such that pair_counts[NOUN][time] == 1244
    """
    count_map = {}
    for pos, word in zip(sequences_A, sequences_B):
        if pos in count_map:
            if word in count_map[pos]:
                count_map[pos][word] += 1
            else:
                count_map[pos][word] = 1
        else:
            count_map[pos] = {word: 1}
    return count_map

def replace_unknown(sequence):
    """Return a copy of the input sequence where each unknown word is replaced
    by the literal string value 'nan'. Pomegranate will ignore these values
    during computation.
    """
    return [w if w in data.training_set.vocab else 'nan' for w in sequence]

def simplify_decoding(X, model):
    """X should be a 1-D sequence of observations for the model to predict"""
    _, state_path = model.viterbi(replace_unknown(X))
    return [state[1].name for state in state_path[1:-1]]  # do not show the start/end state predictions

def accuracy(X, Y, model):
    """Calculate the prediction accuracy by using the model to decode each sequence
    in the input X and comparing the prediction with the true labels in Y.

    The X should be an array whose first dimension is the number of sentences to test,
    and each element of the array should be an iterable of the words in the sequence.
    The arrays X and Y should have the exact same shape.

    X = [("See", "Spot", "run"), ("Run", "Spot", "run", "fast"), ...]
    Y = [(), (), ...]
    """
    correct = total_predictions = 0
    for observations, actual_tags in zip(X, Y):

        # The model.viterbi call in simplify_decoding will return None if the HMM
        # raises an error (for example, if a test sentence contains a word that
        # is out of vocabulary for the training set). Any exception counts the
        # full sentence as an error (which makes this a conservative estimate).
        try:
            most_likely_tags = simplify_decoding(observations, model)
            correct += sum(p == t for p, t in zip(most_likely_tags, actual_tags))
        except:
            pass
        total_predictions += len(observations)
    return correct / total_predictions

#Bigram counts
def bigram_counts(sequences):
    """Return a dictionary keyed to each unique PAIR of values in the input sequences
    list that counts the number of occurrences of pair in the sequences list. The input
    should be a 2-dimensional array.

    For example, if the pair of tags (NOUN, VERB) appear 61582 times, then you should
    return a dictionary such that your_bigram_counts[(NOUN, VERB)] == 61582
    """
    dictionary = {}
    for seq in sequences:
        for i in range(0, len(seq) - 1):
            if (seq[i], seq[i + 1]) not in dictionary:
                dictionary[(seq[i], seq[i + 1])] = 0
            else:
                dictionary[(seq[i], seq[i + 1])] += 1

    return dictionary

# Sequence ending counts
def ending_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequences list
    that counts the number of occurrences where that value is at the end of
    a sequence.

    For example, if 18 sequences end with DET, then you should return a
    dictionary such that your_starting_counts[DET] == 18
    """
    dictionary = {}
    for seq in sequences:
        if seq[-1] not in dictionary:
            dictionary[seq[-1]] = 0
        else:
            dictionary[seq[-1]] += 1

    return dictionary

def unigram_counts(sequences):
    """Return a dictionary keyed to each unique value in the input sequence list that
    counts the number of occurrences of the value in the sequences list. The sequences
    collection should be a 2-dimensional array.

    For example, if the tag NOUN appears 275558 times over all the input sequences,
    then you should return a dictionary such that your_unigram_counts[NOUN] == 275558.
    """
    count_map = Counter(list(sequences)[1])
    return count_map


def print_data_stats(data):
    print("There are {} sentences in the corpus.".format(len(data)))
    print("There are {} sentences in the training set.".format(len(data.training_set)))
    print("There are {} sentences in the testing set.".format(len(data.testing_set)))

    assert len(data) == len(data.training_set) + len(data.testing_set),        "The number of sentences in the training set + testing set should sum to the number of sentences in the corpus"
    key = 'b100-38532'
    print("Sentence: {}".format(key))
    print("words:\n\t{!s}".format(data.sentences[key].words))
    print("tags:\n\t{!s}".format(data.sentences[key].tags))
    print("There are a total of {} samples of {} unique words in the corpus."
          .format(data.N, len(data.vocab)))
    print("There are {} samples of {} unique words in the training set."
          .format(data.training_set.N, len(data.training_set.vocab)))
    print("There are {} samples of {} unique words in the testing set."
          .format(data.testing_set.N, len(data.testing_set.vocab)))
    print("There are {} words in the test set that are missing in the training set."
          .format(len(data.testing_set.vocab - data.training_set.vocab)))

    assert data.N == data.training_set.N + data.testing_set.N,        "The number of training + test samples should sum to the total number of samples"
    # accessing words with Dataset.X and tags with Dataset.Y
    for i in range(2):
        print("Sentence {}:".format(i + 1), data.X[i])
        print()
        print("Labels {}:".format(i + 1), data.Y[i])
        print()

    # use Dataset.stream() (word, tag) samples for the entire corpus
    print("\nStream (word, tag) pairs:\n")
    for i, pair in enumerate(data.stream()):
        print("\t", pair)
        if i > 5: break


# Most frequent class Tagger
# Create a lookup table mfc_table where mfc_table[word] contains the tag label most frequently assigned to that word
def train_mfc_model(data):
    # Calculate C(t_i, w_i)
    emission_counts = pair_counts(*list(zip(*data.training_set.stream()))[::-1])

    assert len(emission_counts) == 12,        "Uh oh. There should be 12 tags in your dictionary."
    assert max(emission_counts["NOUN"], key=emission_counts["NOUN"].get) == 'time',        "Hmmm...'time' is expected to be the most common NOUN."
    HTML('<div class="alert alert-block alert-success">Your emission counts look good!</div>')

    word_counts = pair_counts(*zip(*data.training_set.stream()))

    mfc_table = {i: max(word_counts[i], key=word_counts[i].get) for i in word_counts}
    mfc_model = MFCTagger(mfc_table) # Create a Most Frequent Class tagger instance

    assert len(mfc_table) == len(data.training_set.vocab), ""
    assert all(k in data.training_set.vocab for k in mfc_table.keys()), ""
    assert sum(int(k not in mfc_table) for k in data.testing_set.vocab) == 5521, ""
    HTML('<div class="alert alert-block alert-success">Your MFC tagger has all the correct words!</div>')

    return mfc_model

def get_preds_and_metrics(data,model):
    # ### Making Predictions with a Model
    # The helper functions provided below interface with Pomegranate network models & the mocked MFCTagger to take advantage of the [missing value](http://pomegranate.readthedocs.io/en/latest/nan.html) functionality in Pomegranate through a simple sequence decoding function. Run these functions, then run the next cell to see some of the predictions made by the MFC tagger.

    for key in data.testing_set.keys[:3]:
        print("Sentence Key: {}\n".format(key))
        print("Predicted labels:\n-----------------")
        print(simplify_decoding(data.sentences[key].words, mfc_model))
        print()
        print("Actual labels:\n--------------")
        print(data.sentences[key].tags)
        print("\n")
    # ### Evaluating Model Accuracy
    #
    # The function below will evaluate the accuracy of the MFC tagger on the collection of all sentences from a text corpus.
    # #### Evaluate the accuracy of the MFC tagger
    # Run the next cell to evaluate the accuracy of the tagger on the training and test corpus.
    mfc_training_acc = accuracy(data.training_set.X, data.training_set.Y, mfc_model)
    print("training accuracy mfc_model: {:.2f}%".format(100 * mfc_training_acc))

    mfc_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, mfc_model)
    print("testing accuracy mfc_model: {:.2f}%".format(100 * mfc_testing_acc))

    assert mfc_training_acc >= 0.955, "Uh oh. Your MFC accuracy on the training set doesn't look right."
    assert mfc_testing_acc >= 0.925, "Uh oh. Your MFC accuracy on the testing set doesn't look right."
    HTML('<div class="alert alert-block alert-success">Your MFC tagger accuracy looks correct!</div>')


def build_hmm_tagger(data):
    # Build HMM tagger
    # The HMM tagger has one hidden state for each possible tag, and parameterized by two distributions: the emission probabilties giving the conditional probability of observing a given **word** from each hidden state, and the transition probabilities giving the conditional probability of moving between **tags** during the sequence.
    #
    # We will also estimate the starting probability distribution (the probability of each **tag** being the first tag in a sequence), and the terminal probability distribution (the probability of each **tag** being the last tag in a sequence).
    #
    # The maximum likelihood estimate of these distributions can be calculated from the frequency counts as described in the following sections where you'll implement functions to count the frequencies, and finally build the model. The HMM model will make predictions according to the formula:
    #
    # $$t_i^n = \underset{t_i^n}{\mathrm{argmax}} \prod_{i=1}^n P(w_i|t_i) P(t_i|t_{i-1})$$
    #
    # Refer to Speech & Language Processing [Chapter 10](https://web.stanford.edu/~jurafsky/slp3/10.pdf) for more information.
    # Unigram counts
    tag_unigrams = unigram_counts(zip(*data.training_set.stream()))

    assert set(
        tag_unigrams.keys()) == data.training_set.tagset, "Uh oh. It looks like your tag counts doesn't include all the tags!"
    assert min(tag_unigrams, key=tag_unigrams.get) == 'X', "Hmmm...'X' is expected to be the least common class"
    assert max(tag_unigrams, key=tag_unigrams.get) == 'NOUN', "Hmmm...'NOUN' is expected to be the most common class"
    HTML('<div class="alert alert-block alert-success">Your tag unigrams look good!</div>')

    tag_bigrams = bigram_counts(data.training_set.Y)

    assert len(tag_bigrams) == 144, "Uh oh. There should be 144 pairs of bigrams (12 tags x 12 tags)"
    assert min(tag_bigrams, key=tag_bigrams.get) in [('X', 'NUM'), (
        'PRON', 'X')], "Hmmm...The least common bigram should be one of ('X', 'NUM') or ('PRON', 'X')."
    assert max(tag_bigrams, key=tag_bigrams.get) in [
        ('DET', 'NOUN')], "Hmmm...('DET', 'NOUN') is expected to be the most common bigram."
    HTML('<div class="alert alert-block alert-success">Your tag bigrams look good!</div>')

    tag_starts = starting_counts(data.training_set.Y)

    assert len(tag_starts) == 12, "Uh oh. There should be 12 tags in your dictionary."
    assert min(tag_starts, key=tag_starts.get) == 'X', "Hmmm...'X' is expected to be the least common starting bigram."
    assert max(tag_starts,
               key=tag_starts.get) == 'DET', "Hmmm...'DET' is expected to be the most common starting bigram."
    HTML('<div class="alert alert-block alert-success">Your starting tag counts look good!</div>')

    tag_ends = ending_counts(data.training_set.Y)

    assert len(tag_ends) == 12, "Uh oh. There should be 12 tags in your dictionary."
    assert min(tag_ends, key=tag_ends.get) in ['X',
                                               'CONJ'], "Hmmm...'X' or 'CONJ' should be the least common ending bigram."
    assert max(tag_ends, key=tag_ends.get) == '.', "Hmmm...'.' is expected to be the most common ending bigram."
    HTML('<div class="alert alert-block alert-success">Your ending tag counts look good!</div>')
    return tag_unigrams, tag_bigrams, tag_ends, tag_starts


def train_hmm_tagger(data):
    # HMM
    # Use the tag unigrams and bigrams calculated above to construct a hidden Markov tagger.
    #
    # - Add one state per tag
    #     - The emission distribution at each state should be estimated with the formula: $P(w|t) = \frac{C(t, w)}{C(t)}$
    # - Add an edge from the starting state `basic_model.start` to each tag
    #     - The transition probability should be estimated with the formula: $P(t|start) = \frac{C(start, t)}{C(start)}$
    # - Add an edge from each tag to the end state `basic_model.end`
    #     - The transition probability should be estimated with the formula: $P(end|t) = \frac{C(t, end)}{C(t)}$
    # - Add an edge between _every_ pair of tags
    #     - The transition probability should be estimated with the formula: $P(t_2|t_1) = \frac{C(t_1, t_2)}{C(t_1)}$
    basic_model = HiddenMarkovModel(name="base-hmm-tagger")

    state_dict = {}
    states = []
    emission_counts = pair_counts(*list(zip(*data.training_set.stream()))[::-1])
    for tag in emission_counts.keys():
        tag_count = tag_unigrams[tag]
        probs = {}
        for w in emission_counts[tag]:
            probs[w] = emission_counts[tag][w] / tag_count
        emission_p = DiscreteDistribution(probs)
        state = State(emission_p, name="" + tag)
        basic_model.add_state(state)
        state_dict[tag] = state

    for tag in tag_starts:
        basic_model.add_transition(basic_model.start, state_dict[tag], tag_starts[tag] / len(data.training_set.Y))
        basic_model.add_transition(state_dict[tag], basic_model.end, tag_ends[tag] / tag_unigrams[tag])

    for (tag1, tag2) in tag_bigrams:
        basic_model.add_transition(state_dict[tag1], state_dict[tag2], tag_bigrams[(tag1, tag2)] / tag_unigrams[tag1])

    # finalize the model
    basic_model.bake()

    assert all(tag in set(s.name for s in basic_model.states) for tag in
               data.training_set.tagset), "Every state in your network should use the name of the associated tag, which must be one of the training set tags."
    assert basic_model.edge_count() == 168, (
            "Your network should have an edge from the start node to each state, one edge between every " +
            "pair of tags (states), and an edge from each state to the end node.")
    HTML('<div class="alert alert-block alert-success">Your HMM network topology looks good!</div>')
    return basic_model

if __name__=="__main__":
    data = Dataset("./data/tags-universal.txt", "./data/brown-universal.txt", train_test_split=0.8)
    print_data_stats(data)
    mfc_model = train_mfc_model(data)
    get_preds_and_metrics(data,mfc_model)


    tag_unigrams,tag_bigrams,tag_ends,tag_starts= build_hmm_tagger(data)



    basic_model = train_hmm_tagger(data)
    hmm_training_acc = accuracy(data.training_set.X, data.training_set.Y, basic_model)
    print("training accuracy basic hmm model: {:.2f}%".format(100 * hmm_training_acc))

    hmm_testing_acc = accuracy(data.testing_set.X, data.testing_set.Y, basic_model)
    print("testing accuracy basic hmm model: {:.2f}%".format(100 * hmm_testing_acc))

    assert hmm_training_acc > 0.97, "Uh oh. Your HMM accuracy on the training set doesn't look right."
    assert hmm_testing_acc > 0.955, "Uh oh. Your HMM accuracy on the testing set doesn't look right."
    HTML('<div class="alert alert-block alert-success">Your HMM tagger accuracy looks correct! Congratulations, you\'ve finished the project.</div>')

    # ### E.g. Decoding sequence with HMM tagger
    for key in data.testing_set.keys[:3]:
        print("Sentence Key: {}\n".format(key))
        print("Predicted labels:\n-----------------")
        print(simplify_decoding(data.sentences[key].words, basic_model))
        print()
        print("Actual labels:\n--------------")
        print(data.sentences[key].tags)
        print("\n")
