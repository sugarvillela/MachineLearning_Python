# With a little background knowledge of Machine Learning and Data
# Mining this is easy to understand.  I left print statements you
# can uncomment to see the step-by-step of the algorithms

import numpy as np
from numpy.linalg import inv

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
    
def main():
    test_linear();
    test_fitPoly();

    
if __name__ == '__main__':
	main()
