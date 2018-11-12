# With a little background knowledge of Machine Learning and Data
# Mining this is easy to understand.  I left print statements you
# can uncomment to see the step-by-step of the algorithms

import numpy as np
from numpy.linalg import inv

#================Functions for linear and non-linear regression===========

def loadsparsedata():#x1 x2 x3 y
    X=np.array( [#cloudy=0, reining=1, sunny=2
        [   1,      2       ],
        [   2,      4.5     ],
        [   3.2,    2.1     ],
        [   3.9,    5.1     ],
        [   5.2,    4.2     ]

    ] );
    Y=np.array( [ 1, 1.5, 0.5, 0.8, 0.75 ] );
    return ( X, Y ) #Y.transpose()
    
def addW0( X ):     #Adds a column of 1's before 
    return np.insert( X, 0, 1, axis=1);
    
def polyize( X ):  
    # This function adds extra columns containing x1^2, x2^2...
    # Also adds a column of 1's at the front
    # Useful for fitting data to a polynomial
    # This one only goes to Xi^2; can be generalized further
    # Also could be modified to pass in a function like sinx etc.
    
    stretchBy=len(X[0]);
    X2=np.append( X, np.zeros( [len(X), stretchBy] ), 1 );
    for i in range( 0, len( X[0] ) ):
        for j in range(0, len(X) ):
            X2[ j,i+len( X[0] ) ]=np.power( X[j, i], 2);#X[j, i]
    return np.insert( X2, 0, 1, axis=1);
            

def minLeastSquares( X, Y ):    # w = (XtX)^-1 * (XtY)
    xt=X.transpose();           # XtX
    xtx=np.dot( xt, X );
    xty=np.dot( xt, Y );        # XyY
    xtxi=inv( xtx );             # (XtX)^-1
    w=np.dot( xtxi, xty );      # w = (XtX)^-1 * (XtY)
    # print( 'X transposed' );    # Print steps
    # print( xt );
    # print( 'xtx' );
    # print( xtx );
    # print( 'xty' );
    # print( xty );
    # print( 'xtxi' );
    # print( xtxi );
    # print( 'w' );
    # print( w );
    return w;
    # w=minLeastSquares( X, Y );
    
def minLeastSquares_ridge( X, Y, lam ):# Includes lambda for regularizing
    xt=X.transpose();           # XtX
    xtx=np.dot( xt, X );
    xty=np.dot( xt, Y );        # XyY
    # Ridge regression added here
    lam=np.dot( np.eye( len( X ) ), lam );# make a diagonal matrix from lambda
    xtx=np.add( xtx, lam );               #add to xtx before inverting
    #resume routine as in minLeastSquares
    xtxi=inv( xtx );             # (XtX)^-1
    w=np.dot( xtxi, xty );      # w = (XtX)^-1 * (XtY)
    return w;
    # w=minLeastSquares( X, Y, 2 ); 

def test_linear():
    (X,Y) = loadsparsedata();
    print( 'X' );
    print( X );
    print( 'Y' );
    print( Y );
    X=addW0( X );
    print( 'X W0 added' );
    print( X );
    w=minLeastSquares( X, Y )
    print( 'w minLeastSquares' );
    print( w );
    
def test_fitPoly():
    (X,Y) = loadsparsedata();
    print( 'X' );
    print( X );
    print( 'Y' );
    print( Y );
    X=polyize( X );
    print( 'X poly' );
    print( X );
    w=minLeastSquares( X, Y )
    print( 'w minLeastSquares' );
    print( w );
    w=minLeastSquares_ridge( X, Y, 1 )
    print( 'w minLeastSquares_ridge' );
    print( w );

#================Perceptron Algorithm=====================================
def wRandom( n, last ):
    # Make w a random vector, length n
    # Last is a zero vector, length n
    # There's a one-in-a-gazillion chance of w=[0,0,0...] 
    # Handle that with a do-while loop
    while True:
        w=np.random.randint( -2, 2, size=n );
        if not np.array_equal( w, last ) :
            return w;

def perceptronData():#x1 x2 x3 y
    X=np.array( [#cloudy=0, reining=1, sunny=2
        [   -0.4,   0.75    ],
        [   0,      -0.5    ],
        [   0.2,    0.45    ], 
        [   0.1,    0.2     ], 
        [   0.5,    -0.1    ],
        [   -1.25,  -0.25   ],
        [   1.25,   0       ],
        [   -0.2,   -0.6    ], 
        [   -0.5,   0.5     ],
    ] );
    Y=np.array( [1,1, -1, -1, -1, 1, -1, 1, 1] ) ;
    return ( X, Y )

def perceptron( X, Y, setw, ada, halt) :
    # Unsure if this function will ever halt
    # Set a 'last' vector and compare after every try: w==last means done
    # Also set a hard 'halt' in case it never finds a solution
    # BTW, w is the vector orthogonal to the decision surface
    last=np.zeros( len( X[0] ) );
    if( setw is None ):
        w=wRandom( len( X[0] ), last ); # np.array( [-1,1,1] );
    else:
        w=setw;
    for h in range(0, halt ):
        for i in range( 0, len( X ) ) :
            # If wt*Xi different sign, change direction of vector w
            if np.sign( np.dot( w, X[i] ) ) != np.sign( Y[i] ):
                nyx=np.dot( X[i], ada*Y[i])
                w=np.add( w, nyx );
		
        if np.array_equal( w, last ) :
            break;
        last=w;
    return w;
    
def test_perceptron():
    (X,Y) = perceptronData();
    print( 'X' );
    print( X );
    print( 'Y' );
    print( Y );
    w=perceptron( X, Y, 1, 10 );
    print( 'perceptron w: -0.6, -0.1' )
    print( w )

#================Gradient Descent=========================================

# Sigmoid maps values to somewhere between 0 and 1
# You can also use ReLU, which is max(0,fx)
def sigmoid( fx ):
    return 1/( 1 + np.exp( fx*-1 ) );

# This is from some in-class slides
# Sochastic means it updates w after every calculation
# Simple, no regularizer
def gradDescent_sochastic( X, Y, w, eta) :
    halt=1;
    for h in range( 0, halt ):
        g=np.zeros( len( w ) );
        print('g:',g);
        for i in range( 0, len( Y ) ) :
            print( 'w=', w );       #This outputs current w to match slide example
            s=np.dot( w, X[i] );    #wtXi
            s=np.dot( Y[i], s );    #yi*wtxi
            s=sigmoid( s );         #sig( yiwtxi )
            s=1-s;                  #(1-pi)
            s*=Y[i];                #(1-pi)yi
            s=np.dot( X[i], s );    #(1-pi)yixi
            s=np.dot( eta, s );     #eta*(1-pi)yixi
            w=np.add( w, s );       #w=w+eta*(1-pi)yixi
    return w;

def exampleData():# From slide set
    X=np.array( [
        [   1,  0.45,   3.25    ], 
        [   1,  -1.08,  2.2     ],
        [   1,  .2,     1.18    ],
        [   1,  -1.18,  .98     ],
        [   1,  -2.49,  3.59    ]
    ] );
    Y=np.array( [1, -1, -1, 1, 1] ) ;
    return ( X, Y )

def test_exampleData(  ):
    (X,Y) = exampleData();
    print( 'X' );
    print( X );
    print( 'Y' );
    print( Y );
    w=np.array( [-1,1,1] );
    eta=0.1;
    w=gradDescent_sochastic( X, Y, w, eta );
    print( 'final w' )
    print( w )
	
#================Below: Same thing but batch with regularizer=============	

# First, some given code for loading train and test files
def loadsparsedata(fn):
    fp = open(fn,"r")
    lines = fp.readlines()
    maxf = 0;
    for line in lines:
        for i in line.split()[1::2]:
            maxf = max(maxf,int(i))
    
    X = np.zeros((len(lines),maxf))
    Y = np.zeros((len(lines)))
    
    for i, line in enumerate(lines):
        values = line.split()
        Y[i] = int(values[0])
        for j,v in zip(values[1::2],values[2::2]):
            X[i,int(j)-1] = int(v)
    
    return X,Y

# Data in file is 0-1: map -1 to 0
def getXY(fn):
    (X,Y) = loadsparsedata(fn);
    # add column of 1s as zeroth feature
    X = np.column_stack((np.ones(X.shape[0]),X)) 
    # change from 0's to -1's
    Y[Y==0] = -1;
    return X,Y

# For deriviative of sum of squared w   
def reg( lam, m, w ):# passing in 1 for m
    w1=np.insert( w[1:], 0, 0 );
    return np.dot( w1, 2*lam/m );

# To save time in dev, set hardHalt low 
# If you've got an hour to kill, set high and wait for condHalt
# Halts after 5,000 to 50,000 iterations of inner loop
def gradDescent_batch( X, Y, w, eta, lam ) :
    hardHalt=100000;
    condHalt=1e-10;
    minL=10000000;
    for h in range( 0, hardHalt ):
        # w only changes at end of batch, so just add regularizer to g now
        g=reg( lam, 1, w );#regularization
        for i in range( 0, len( Y ) ) :
            #Each step Barney-style with comment
            s=np.dot( w, X[i] );    # wtXi
            s=np.dot( Y[i], s );    # yi*wtxi
            s=sigmoid( s );         # sig( yiwtxi )
            s=s*(1-s);              # (1-pi)
            s*=Y[i];                # (1-pi)yi
            s=np.dot( X[i], s );    # (1-pi)yixi
            s=np.dot( s, -1 );      # -(1-pi)yixi
            g=np.add( g, s );
        g=np.dot( g, 1/len(Y) );# 1/m( -(1-pi)yixi )
        g=np.dot( eta, g );	# n/m( -(1-pi)yixi ) n being eta
        w=np.subtract( w, g );	# w-=g
        lensq=np.dot( g, g )	# lensq = how much change is happening each iteration
        if h and lensq<condHalt:# Quit when slope gets near zero
            print( 'condHalt @ h=', h );
            return w; 
        if lensq<minL:		# Just for display, keep track of smallest change so far
            minL=lensq; 
        if h%10000==0:		# Output something periodically to see it working
            print( 'minL', lam, h, minL );
    return w;

# this sets up initial w and eta and runs gradientDescent
def learnlogreg(X,Y,lam):
    (m,n) = X.shape;
    w = np.zeros((n));
    eta = np.sqrt(2.5/lam);
    return gradDescent_batch( X, Y, w, eta, lam );

# This code is given
def linearerror(X,Y,w):
    # returns error *rate* for linear classifier with coefficients w
    m = Y.shape[0]
    predy = X.dot(w)
    err = (Y[predy>=0]<0.5).sum() + (Y[predy<0]>=0.5).sum()
    return err/m

# This runs gradDescent_batch on the training files
def testGradDescent_batch():
    # getXY gets the data, adds the offset column to X and fixes the 0's in Y
    (X,Y) = getXY("spamtrain.txt");
    lam=.001;
    w = learnlogreg(X,Y,lam);
    print( 'err=',linearerror(X,Y,w) );

#================Use batch gradDescent with cross-validation==============

import matplotlib.pyplot as plt
from IPython.display import display, clear_output

# This splits training file into div-size pieces and finds best lambda
def crossval( trainFile, lambdas, div ):
    if( div and div-1 ):        #assert non-zero denominator
        denom=div*(div-1);          # for avg over number of tests
    else:
        return None;
        
    (X,Y) = getXY( trainFile );
    Xcross=np.split( X, div );#returns list
    Ycross=np.split( Y, div );
    errs=np.zeros( len( lambdas ) );
    bestW=0;
    bestLam=0;
    minErr=1000000;
    h=0;
    
    for lam in lambdas:
        print('crossVal: lambda =', lam );
        for V, VY in zip( Xcross, Ycross ): # the splits of X, Y
            w=learnlogreg( V, VY, lam );
            curErr=0;
            for T, TY in zip( Xcross, Ycross ):# splits of X, Y that are not V
                if V is not T:
                    curErr+=linearerror( T, TY, w );
            if( curErr<minErr ):
                minErr=curErr;
                bestW=w;
                bestLam=lam;
            errs[h]+=curErr;
        errs[h]/=denom;
        h+=1;
    return ( bestLam, bestW, errs );

# This finds best lambda, training with train file and testing on test file
def trainTest( trainFile, testFile, lambdas ):
    (trainX,trainY) = getXY("spamtrain.txt");
    (testX,testY) = getXY("spamtest.txt");
    errs=np.zeros( len( lambdas ) );
    bestW=0;
    bestLam=0;
    minErr=1000000;
    h=0;
    for lam in lambdas:
        print('Test: lambda =', lam );
        w=learnlogreg( trainX, trainY, lam );
        errs[h]=linearerror( testX, testY, w );
        if( errs[h]<minErr ):
            minErr=errs[h];
            bestW=w;
            bestLam=lam;
        h+=1;
    return ( bestLam, bestW, errs );

# This is a hack to make lambdas fit into graph, totally avoidable 
# if I knew more about matplotlib
def lamToTrunc( lambdas ):
    arr=np.array2string( lambdas )[1:-1].split(' ');
    for i in range( 0, len(arr) ):
        arr[i]=arr[i].split('.')[0]+'e'+arr[i].split('e')[1];
    return arr;

def dispLamVsErr( lambdas, valErrs, testErrs ):
    xtix=lamToTrunc( lambdas );
    x_plot = range(len(xtix));
    plt.plot( x_plot, valErrs, label='V' );#'o', 
    plt.plot( x_plot, testErrs, label='T' );
    plt.xticks(x_plot, xtix );
    plt.xlabel('Lambda');
    plt.ylabel('Error');
    plt.show();
    
def test_crossval():
    lambdas=np.logspace(-3,2,10);#-3 to 2 is log range, 10 is size of returned list
    print ('Running Crossval...');
    ( bestLam, bestW, valErrs )=crossval( "spamtrain.txt",lambdas, 5 )
    print('bestLam =', bestLam );
    print('valErrs =', valErrs );
    
    print ('Running Test...');
    ( bestLam, bestW, testErrs )=trainTest( "spamtrain.txt", "spamtest.txt", lambdas );
    print('bestLam =', bestLam );
    print('testErrs =', testErrs );
    print( );
    dispLamVsErr( lambdas, valErrs, testErrs );
	

def main():
    test_linear();
    test_fitPoly();
    test_gradDescent();
    test_gradDescent_batch();
    test_crossval();
    
if __name__ == '__main__':
	main()
