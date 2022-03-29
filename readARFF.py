import sys
import pickle

### this is a modified version of readARFF that, for the name of each attribute,
### also keeps track of its order. This will let us map attributes to columns 
### in the data.  
### read in data from an ARFF file and return the following data structures:
### A dict that maps attribute names to a tuple with two components.
### the first component is the columnn corresponding to this attribute.
### the second is a list containing either:
###    - possible values
###    - the string 'string'
###    - the string 'numeric'

def readArff(filehandle) :
    attributes = {}
    data = []
    relation = ""
    ### remove all commented and blank lines.
    lines = [line for line in filehandle.readlines() if not line.startswith('%') and len(line) > 1]
    ### get all attribute lines
    attribLines = [line for line in lines if line.startswith('@attribute')]
    i = 0
    for line in attribLines :
        chunks = line.split(' ',2)    
        if chunks[2].startswith('{') : ### this is a nominal attribute
            attributes[chunks[1]] = (i,[item.strip() for item in chunks[2].strip('\n{}').split(',')])
        else : ## this is string or numeric
            attributes[chunks[1]] = (i, chunks[2])
        i+=1
    
    ### data will be lines that don't begin with a '@'
    dataLines = [line for line in lines if not line.startswith('@')]
    for line in dataLines :
        data.append(line.strip().split(','))
    return (attributes, data)


### takes as input the dictionary of attributes and returns a list of
### the attributes (only, no values) in the correct order. 
def getAttrList(attrs) :
    alist = [(attrs[item][0], item) for item in attrs]
    alist.sort()
    return [item[1] for item in alist]


### Compute ZeroR - that is, the most common data classification without 
### examining any of the attributes. Return the most comon classification.
def computeZeroR(attributes, data) :
    max = 0
    maxItem = ''
    classes = [item[-1] for item in data]
    vals = set(classes)
    for item in vals :
        if classes.count(item) > max :
            max = classes.count(item)
            maxItem = item
    return maxItem
    


### Usage: readARFF {--pfile=outfile} infile
### If --pfile=outfile, pickle and store the results in outfile. Otherwise, 
### print them to standard out. Your code should also call computeZeroR and 
### print out the results.

if __name__ == '__main__' :
    if len(sys.argv) < 2 :
        print("Usage: readARFF {--pfile=outfile} infile")
        sys.exit(-1)
    fname = sys.argv[-1]
    (attrs, data) = readArff(open(fname))
    print("Most common classification is: ", computeZeroR(attrs,data))
    if sys.argv[1].startswith("--pfile") :
        ofile = sys.argv[1].split('=')[1]
        fh = open(ofile, 'w')
        pickle.dump(attrs, fh)
        pickle.dump(data, fh)
    else :
        print ("Attributes: " + str(attrs))
        print ("Data: " +  str(data))
