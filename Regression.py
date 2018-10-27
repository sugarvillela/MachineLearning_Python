# With a little background knowledge of Machine Learning and Data
# Mining this is easy to understand.  I left print statements you
# can uncomment to see the step-by-step of the algorithms

import numpy as np
from numpy.linalg import inv

# Functions for linear and non-linear regression

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

# Perceptron Algorithm

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
    #test_perceptron();

# Gradient Descent (batch method)
def gradientData():#x1 x2 x3 y
    X=np.array( [#cloudy=0, reining=1, sunny=2
        [   1,  0.45,   3.25    ], 
        [   1,  -1.08,  2.2     ],
        [   1,  .2,     1.18    ],
        [   1,  -1.18,  .98     ],
        [   1,  -2.49,  3.59    ]
    ] );
    Y=np.array( [1, -1, -1, 1, 1] ) ;
    return ( X, Y )

def sigmoid( fx ):
    return 1/( 1 + np.exp( fx*-1 ) );
    
def sigPrime( fx ):
    sig=sigmoid( fx );
    return (1-sig);
    
def gradDescent_sochastic( X, Y, w, eta) :
    halt=1;
    for h in range( 0, halt ):
        g=np.zeros( len( w ) );
        print('g:',g);
        for i in range( 0, len( Y ) ) :
            #print( 'w=', w );
            s=np.dot( w, X[i] );    #wtXi
            s=np.dot( Y[i], s );    #yi*wtxi
            s=sigmoid( s );         #sig( yiwtxi )
            s=1-s;                  #(1-pi)
            s*=Y[i];                #(1-pi)yi
            s=np.dot( X[i], s );    #(1-pi)yixi
            s=np.dot( eta, s );     #eta*(1-pi)yixi
            w=np.add( w, s );       #w=w+eta*(1-pi)yixi
    return w;
    
def gradDescent_batch( X, Y, w, eta) :
    halt=1;
    for h in range( 0, halt ):
        g=np.zeros( len( w ) );
        print('g:',g);
        for i in range( 0, len( Y ) ) :
            print( 'g=', g );
            s=np.dot( w, X[i] );    #wtXi
            s=np.dot( Y[i], s );    #yi*wtxi
            s=sigmoid( s );         #sig( yiwtxi )
            s=1-s;                  #(1-pi)
            s*=Y[i];                #(1-pi)yi
            s=np.dot( X[i], s );    #(1-pi)yixi
            s=np.dot( s, -1 );      #-(1-pi)yixi
            w=np.add( w, s );       #w=w+eta*(1-pi)yixi
        g=np.dot( g, 1/len(Y) );    #1/m( g )
        g=np.dot( eta, g );
        w=np.subtract( w, g );
    return w;
    
def test_gradDescent():
    (X,Y) = exampleData();
    print( 'X' );
    print( X );
    print( 'Y' );
    print( Y );
    w=np.array( [-1,1,1] );
    eta=0.1;
    w=gradDescent_batch( X, Y, w, eta );
    print( 'final w' )
    print( w )

def main():
    test_linear();
    test_fitPoly();
    test_gradDescent();

    
if __name__ == '__main__':
	main()
