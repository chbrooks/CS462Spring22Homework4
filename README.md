
## Decision Trees and Hidden Markov Models

#### Due: Tuesday,April 19, Start of class.

#### Part 1: Probabilities

**(5 points)** Show how to derive Bayes' rule from the product rule.

**(10 points)** Consider two medical tests, A and B, for a disease. 
Test A is 95% effective at recognizing the disease when it is present, 
but has a 10% false positive rate (indicating that the disease is present, when it is not). 
Test B is 90% effective at recognizing the disease, but has a 5% false positive rate. 
The two tests use independent methods of identifying the disease. 
1% of the population has this disease.

Suppose that we are particularly interested in minimizing false positives - that is, we do not want 
cases in which someone who does not have the disease to test positive.

Which test would we prefer? Justify your answer using Bayes' rule.


##### Part 2: Learning decision trees
In this assignment, you'll implement the classic decision tree algorithm.
The amount of code you need to write is not huge, but there
are some mental hurdles you'll need to cross to completely understand what's going on.

I have provided some skeleton code to get you started and guide you
through the implementation. You are welcome to make any changes that
you like, but please think carefully before doing so; this is designed
to help make this easy to implement.

Datasets:

There are three datasets included with the assignment.

  - Toy datasets:
    - the tennis dataset
    - the restaurant dataset.
    
    These are both useful for testing your code; they're small, and
    you know what the correct answers are.
  
      - Breast Cancer data. We used this dataset in assignment 1. This dataset contains medical records
      from a large number of women who have had breast cancer. Based
      on their characteristics, we would like to predict whether they will have a recurrence event. 
.

   These files are in a format known as [ARFF](https://datahub.io/blog/attribute-relation-file-format-arff), which we saw in assignment 1. 
   An ARFF file consists of three sections. The first is the comments, which begin with a '%'. The second is the @relation section, which 
   describes each of the variables and the values they can take on. The third is the data itself, with each row representing one instance.
   I've provided you with a file called readARFF.py. to process this. If you want to resue your assignment 1 code, please do!

   readARFF contains three functions:
   - readArff, which takes as input a filehandle and returns two items:
     - an attribute dictionary which makes an attribute name to a tuple indicating the column that attribute represents in the dataset and a list of the possible values it can take on.)
     - a list of lists containing the data itself. 
     
  - getAttrList. This takes as input the attribute dictionary and returns a list of the attribute names in the same order as the columns they represent.
  - ZeroR. Our old friend. Returns the most likely classification in a dataset. You may find it useful as a comparator, or in cases where you are unable to split your dataset.
  
A hint: list comprehensions are very helpful for this
   assignment. Often, you'll need to pull out one or more columns from
   the data. So, for example, to get a list containing only the third
   column in a dataset where the last element is equal to some item
   'x', you could do: 
   <pre>
third = [d[2] for d in data if d[-1] == 'x']
</pre>

Those of you who are familiar with pandas and would like to use dataframes for this assignment are welcome to do so; this is not required, though.

First, we'll implement the basic decision tree algorithm. 

- **(10 points)** The decision tree is easiest to build in a bottom-up fashion. To
   begin, we'll need a method to compute entropy. it should take 
   as input a list, such as ['weak', 'strong',
   'weak', 'weak'] and return a float indicating the entropy in this
   data. I've provided a function stub for you. 

- Next, we'll want to compute remainder. This will tell us, for a
   given attribute, how much information will remain if we choose to
   split on this attribute. I've written this one for you. 


- **(10 points)** Once we know how to compute remainders, we need to be able to
   select an attribute. To do this, we just compute the remainder for
   each attribute and choose the one with the smallest
   remainder. (this will maximize information gain.) The function
   selectAttribute should take as input a list of lists, with each
   list being an instance. I've provided a stub for you.

Now we're now ready to think about building a tree. A tree is a
   recursive data structure which consists of a parent node that has
   links to child nodes. I've provided a TreeNode class for you that
   does this. (you don't need a separate Tree class.)

The TreeNode has the following data members:
- attribute: for non-leaf nodes, this will indicate which attribute
   this node tests. For leaves, it is empty. 
- value. For leaf nodes, this indicates the classification at
   this leaf. For non-leaf nodes, it is empty. 
- children. This is a dictionary that maps values of the
   attribute being tested at this node to the appropriate child, which
   is also a TreeNode. 
 
It also has methods to print itself and to test whether it is a
   leaf.
 
**(10 points)** So we need a method that can build a tree. We will call this
   makeTree. It should take as input a dataset, a list of attribute
   names, the attribute dictionary, and a default value. It should work as follows:
 - If the dataset contains zero entropy, we are done. Create a
   leaf node with value equal to the data's classification and return
   it. 
 - If the dataset is empty, we have no data for this attribute
   value. Create a leaf node with the value set to the default value
   and return it. 
 - Otherwise, we have a non-leaf node. Use selectAttribute to
   find the attribute that maximizes gain. Then, remove that column
   for the dataset and the list of attributes and, for each value of
   that attribute, call makeTree with the appropriate subset of the
   data and add the TreeNode that is returned to the children, then
   return the TreeNode.
   
  - **(10 points)** Now we know how to build a tree. We need to use it, though. To
   do this, you should implement the classify() method in
   TreeNode. classify should take as input the data instance to be
   classified and our attribute dictionary.
   
    This method is also recursive. If we are at a leaf, return the
   value of that leaf. Otherwise, check which attribute this node
   tests and follow the appropriate child. If there is no child for
   this value, return a default value.
   
Congratulations! You now have a working decision tree. Test it
   out on the toy datasets. You might find it helpful to build a
   better printTree method, although this is not required.

- **(10 points)** Please add a submission.py that demonstrates your decision tree. It should have the option to run your tree on either of the toy datasets, or to 
run on the breast cancer data using five-fold cross-validation, displaying the following statistics:
  - Precision 
  - Accuracy 
  - F1
  
  If you would like to use sklearn to manage the cross-validation, you may.
 

#### Part 3: Hidden Markov Models.
(Note: this is derived from an assignment in AAAI's Model Assignments workshop)

In this assignment you'll be implementing two algorithms associated with Hidden Markov Models.

You'll be building off of the code presented in HMM.py.
There's also some included data to use. 

The first set of files are .trans files. They contain the transition probabilities.
two_english models the transition between 'C' for Consonant and 'V' for Vowel in English. 
partofspeech.browntags.trained models transitions between parts of speech in the Brown corpus. 

The second set of files are .emit files. These contain the probability of emitting a particular output from that 
state. These are learned from data, and so contain errors (especially two_english).


**(10 points)**. Use the included code to implement load. Use two_english as a sample file to work with.
You should be able to do:

<pre>
    model = HMM()
    model.load('two_english')
</pre>

You should store the transitions and emissions as dictionaries of dictionaries. 
e.g. {'#': {'C': 0.814506898514, 'V': 0.185493101486},
    'C': {'C': 0.625840873591, 'V': 0.374159126409},
    'V': {'C': 0.603126993184, 'V': 0.396873006816}}

**(10 points)** Implement generate. It should take an integer n, and return a random observation of length n. To generate this, start in the initial state and repeatedly select successor states at random, using the probability as a weight, 
and then select an emission, again using the probability as a weight. You may find either numpy.random.choice or random.choices very helpful here.


You should be able to run it with the pre-trained probabilities for the Brown corpus, like so:
<pre>
python hmm.py partofspeech.browntags.trained --generate 20
</pre>

which generates 20 random observations.

Here are two sample observations:
<pre>
DET ADJ . ADV ADJ VERB ADP DET ADJ NOUN VERB ADJ NOUN
the semi-catatonic , quite several must of an western bridge cannot spectacular analyses
DET NOUN ADP NOUN CONJ DET VERB DET NOUN NOUN NOUN ADP DET NOUN
whose light for wall and the learned the hull postmaster trash in his peters
</pre>

**(15 points)** Next, implement Viterbi. This tells us, for a sequence of observations, the most likely sequence of states. You should be able to run this like so:

<pre>
python hmm.py partofspeech.browntags.trained --viterbi ambiguous_sents.obs
</pre>

This uses the HMM parameters in partofspeech.browntags.trained.{trans,emit} to compute the best sequence of part-of-speech tags for each sentence in ambiguous_sents.obs, and writes it to ambiguous_sents.out.

You might find it helpful to use a numpy array to hold the matrix.

Compare the output file to ambiguous_sents.tagged.obs. 

**(686 students only)** Please read [this article](https://www.theatlantic.com/magazine/archive/2013/11/the-man-who-would-teach-machines-to-think/309529/)
about Douglas Hofstadter, which also serves as a nice summary of the history of AI and the debates over 
the value of developing machines that think like humans. (As an aside: If you have not read Hofastadter's book [Godel, Escher, Bach](https://en.wikipedia.org/wiki/G%C3%B6del,_Escher,_Bach), I strongly recommend it.)

Prepare a summary or critique of this article that addresses the following questions:

- Hofstadter is particularly interested in understanding the way humans think. What sorts of reasoning mechanisms does he study?
- The article includes a quote from our text: “The quest for ‘artificial flight’ succeeded when the Wright brothers and others stopped imitating birds and started … learning about aerodynamics,” What does this mean? Why is it relevant to AI?
- What was Candide? Why did it change the way we thought about machine translation? 
- The article also contains a quote from the last chapter of AIMA: perhaps AI has become too much like the man who tries to get to the moon by climbing a tree: “One can report steady progress, all the way to the top of the tree.” What does this mean? How does it 
  relate to Candide and the ways in which big data and machine learning have changed AI?
