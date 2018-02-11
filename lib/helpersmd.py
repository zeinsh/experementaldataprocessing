import numpy as np

# File Functions
def write_to_file(report,filename):
    """output report to text file:
       parameters:
	   report: text to output
	   filename: report will be directed to output/filename
       nothing to return
    """
    f=open("output/"+filename,"w")
    f.write(str(report))
    f.close()

def readFromDat(filepath,dtype,count=-1):
    """read .dat file 

	parameters:
	    filename: str, .dat file path
	    dtype:  type of binary values for example:[float,int,np.float32,...]
	    count: how many values to return

	return:
	    numpy array contains bin values returned from .dat file

    """
    myarray = np.fromfile(filepath,dtype=dtype,count=count)
    return myarray

# Series Generating functions
def getRandomND(N): #Normal distribution
    """ get N random values with normal distribution

	parameters:
	    N: number of random values to return

    """
    return (np.random.normal(size=(N,1)))
def getMyRandom(N):
    """ self random generating function

	parameters:
            N: number of random values to return

    """
    rn=getRandomND(N)
    return np.sin(rn)

