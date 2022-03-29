import sys, math, re
import pickle
import readARFF

### takes as input a list of attribute values. Returns a float
### indicating the entropy in this data.

def entropy(data) :



### remainder - this is the amount of entropy left in the data after
### we split on a particular attribute. Let's assume the input is of
### the form: [(value1, class1), (value2, class2), ..., (valuen,
### classn)] I've written this for you.

def remainder(data) :
    possibleValues = set([item[0] for item in data])
    r = 0.0
    for value in possibleValues :
        c = [item[0] for item in data].count(value)  
        r += (float(c) / len(data) ) * entropy([item[1] for item in
                                                data if item[0] == value])
    return r


### choose the index of the attribute in the current dataset that
### minimizes the remainder. 
### data is in the form [[a1, a2, ..., c1], [b1,b2,...,c2], ... ]
### where the a's are attribute values and the c's are classifications.
### and attributes is a list [a1,a2,...,an] of corresponding attribute values

def selectAttribute(data, attributes) :


### a TreeNode is an object that has either:
### 1. An attribute to be tested and a set of children; one for each possible 
### value of the attribute.
### 2. A value (if it is a leaf in a tree)

class TreeNode :
    def __init__(self, attribute, value) :
        self.attribute = attribute
        self.value = value
        self.children = {}

    def __repr__(self) :
        if self.attribute :
            return self.attribute
        else :
            return self.value

    ### a node with no children is a leaf
    def isLeaf(self) :
        return self.children == {}

    ### the input will be:
    ### data - an object to classify - [v1, v2, ..., vn]
    ### the attribute dictionary

    def classify(self, data, attributes) :


### a tree is simply a data structure composed of nodes. The root of the tree 
### is itself a node, so we don't need a separate 'Tree' class. We
### just need a function that takes in a dataset and our attribute dictionary,
### builds a tree, and returns the root node.
### makeTree is a recursive function. Our base case is that our
### dataset has entropy 0 - no further tests have to be made. There
### are two other degenerate base cases: when there is no more data to
### use, and when we have no data for a particular value. In this case
### we use either default value or majority value.
### The recursive step is to select the attribute that most increases
### the gain and split on that.

### assume: input looks like this:
### dataset: [[v1, v2, ..., vn, c1], [v1,v2, ..., c2] ... ]
### alist: [a1,a2,...,an]
### attributes: {a1 : [v1,v2,v3],a2:[v4,v5],...}

def makeTree(dataset, alist, attributes, defaultValue) :


