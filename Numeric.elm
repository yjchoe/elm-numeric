module Numeric where

{-| Numeric.js in Elm
    Fast large-scale matrix and vector operations in Elm

    numeric.js (http://www.numericjs.com/) by Sebastien Loisel
    Edited for the Elm programming language by Yo Joong Choe

-}

import Native.Numeric
import List
import String

{-| Type `Ndarray` is either a multi-dimensional JavaScript array
    or a JavaScript float (as a result of vector-vector dot product).
    In the context of Elm, `Ndarray` is considered as an internal sructure,
    and the final output is expected to be presented using `fromArray`.
-}
type Ndarray = Ndarray

{-| For representation purposes, we provide a way to convert
    the internal structure `Ndarray` to/from a nested `List` in Elm.
    We call this `Ndlist`, which is a type of a `List`.

    Converting to/from `Ndlist` is *slow*. This type should only be
    used for viewing storing/viewing the final result in an Elm-friendly type.
-}

{-| Initializes an array of dimensions specified by the `List` of `Int`s. 
    All entries are initialized to zero. 

    ndarray [3,2] == fromList [[0, 0], 
                               [0, 0], 
                               [0, 0]]
-}
ndarray : List Int -> Ndarray
ndarray = Native.Numeric.ndarray

fromList : List a -> Ndarray
fromList = Native.Numeric.fromList

fromCSV : String -> Ndarray
fromCSV = Native.Numeric.fromCSV

vec : Int -> Ndarray
vec = Native.Numeric.vec

mat : Int -> Int -> Ndarray
mat = Native.Numeric.mat

rep : Ndarray -> Float -> Ndarray
rep = Native.Numeric.rep

zeros : List Int -> Ndarray
zeros = Native.Numeric.zeros

ones : List Int -> Ndarray
ones = Native.Numeric.ones

{-| Initializes an array of dimensions specified by the `List` of `Int`s. 
    All entries are initialized from a uniform distribution between 0 and 1. 
    The following example contains one possible result. 

    random [3,2] == fromList [[0.6078, 0.6633], 
                              [0.5165, 0.3208], 
                              [0.1039, 0.8389]]
-}
random : List Int -> Ndarray
random = Native.Numeric.random

identity : Int -> Ndarray
identity = Native.Numeric.identity

eye : Int -> Ndarray
eye = Native.Numeric.identity

{-| Slicing operations -}

getBlock : List Int -> List Int -> Ndarray -> Ndarray
getBlock = Native.Numeric.getBlock

{-| Unary entrywise operations -}

abs : Ndarray -> Ndarray
abs = Native.Numeric.abs

acos : Ndarray -> Ndarray
acos = Native.Numeric.acos

asin : Ndarray -> Ndarray
asin = Native.Numeric.asin

atan : Ndarray -> Ndarray
atan = Native.Numeric.atan

ceil : Ndarray -> Ndarray
ceil = Native.Numeric.ceil

ceiling : Ndarray -> Ndarray
ceiling = Native.Numeric.ceil

conj : Ndarray -> Ndarray
conj = Native.Numeric.conj

conjugate : Ndarray -> Ndarray
conjugate = Native.Numeric.conj

cos : Ndarray -> Ndarray
cos = Native.Numeric.cos

exp : Ndarray -> Ndarray
exp = Native.Numeric.exp

floor : Ndarray -> Ndarray
floor = Native.Numeric.floor

log : Ndarray -> Ndarray
log = Native.Numeric.log

neg : Ndarray -> Ndarray
neg = Native.Numeric.neg

pow : Ndarray -> number -> Ndarray
pow = Native.Numeric.pow

round : Ndarray -> Ndarray
round = Native.Numeric.round

sin : Ndarray -> Ndarray
sin = Native.Numeric.sin

sqrt : Ndarray -> Ndarray
sqrt = Native.Numeric.sqrt


{-| Binary entrywise operations -}

atan2 : Ndarray -> Ndarray -> Ndarray
atan2 = Native.Numeric.atan2

add : Ndarray -> Ndarray -> Ndarray
add = Native.Numeric.add

div : Ndarray -> Ndarray -> Ndarray
div = Native.Numeric.div

mod : Ndarray -> Ndarray -> Ndarray
mod = Native.Numeric.mod

mul : Ndarray -> Ndarray -> Ndarray
mul = Native.Numeric.mul

sub : Ndarray -> Ndarray -> Ndarray
sub = Native.Numeric.sub

{-| Matrix/Tensor routines -}

{-|
normalize : Ndarray -> Ndarray
normalize x = 
-}

det : Ndarray -> Float
det = Native.Numeric.det

diag : Ndarray -> Ndarray
diag = Native.Numeric.diag

dim : Ndarray -> List Int
dim = Native.Numeric.dim >> toList

{-| A polymorphic dot product function.
    Functionalities include matrix-matrix, matrix-vector, and vector-matrix 
    multiplications, as well as the standard vector-vector dot product
    and scalar-vector multiplication.

    Note that, unlike other results, the result of vector-vector dot product 
    is a JavaScript float that is typed in Elm as an `Ndarray`. While such
    polymorphism is useful in imperative languages, here we do also provide 
    a separate function `dotVV` for the vector-vector dot product whose 
    return type is `Float`.  
-}
dot : Ndarray -> Ndarray -> Ndarray
dot = Native.Numeric.dot

dotVV : Ndarray -> Ndarray -> Float
dotVV = Native.Numeric.dotVV 

{-| Fast eigenvalue/eigenvector computation using the Francis QR algorithm.
    Given a square matrix, `eig` returns a record with the vector of
    eigenvalues (`lambda`) and the matrix of eigenvectors (`v`).

    Because eigenvalues and eigenvectors are complex in general --
    even when the input matrix is real -- the values of `lambda` and `v` are
    again records of the form `{ x: Ndarray, y: Ndarray }`, where `x` and `y`
    are the real and imaginary parts of the result, respectively.

    eig (identity 5) == { lambda = { x = ones [5], y = zeros [5] }
                        , v      = { x = identity 5, y = zeros [5,5] } 
                        }
-}
eig : Ndarray -> { lambda: Ndarray, v: Ndarray }
eig = Native.Numeric.eig

{-| Returns the diagonal of a multi-dimensional array.

    getDiag (fromList [[1,2,3],[4,5,6]]) == fromList [1,5]
-}
getDiag : Ndarray -> Ndarray
getDiag = Native.Numeric.getDiag

inv : Ndarray -> Ndarray
inv = Native.Numeric.inv

{-| `linspace a b n` gives `n` evenly spaced numbers from `a` to `b`.

    linspace 0.0 1.0 6 == fromList [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
-}
linspace : Float -> Float -> Int -> Ndarray
linspace = Native.Numeric.linspace

norm2 : Ndarray -> Float
norm2 = Native.Numeric.norm2

norm2sq : Ndarray -> Float
norm2sq = Native.Numeric.norm2sq

norminf : Ndarray -> Float
norminf = Native.Numeric.norminf

{-| `solve` gives the solution to the linear system Ax=b.
    That is, `solve a b == x`.

    solve (identity 5) (vec 5) == vec 5
-}
solve : Ndarray -> Ndarray -> Ndarray
solve = Native.Numeric.solve

{-| Performs singular value decomposition (SVD) of a rectangular matrix a,
    using the Golub-Reinsch method (1970).
    Returns a record containing the matrix of left singular vectors (u),
    the vector of singular values (s), and the matrix of right singular
    vectors (v).
-}
svd : Ndarray -> { u: Ndarray, s: Ndarray, v: Ndarray }
svd = Native.Numeric.svd

tr : Ndarray -> Ndarray
tr = Native.Numeric.tr

transpose : Ndarray -> Ndarray
transpose = Native.Numeric.tr

{-| Optimization [TO BE IMPLEMENTED] -}

{-| Inputs are, in general, records, as many parameters are specifications
    of the optimization problem.

defaultOption : 

gradient(f,x)

gradient : {} -> Ndarray
gradient = Native.Numeric.gradient

uncmin(f,x0,tol,gradient,maxit,callback,options)

uncmin : {} -> Ndarray
uncmin = Native.Numeric.uncmin

solveLP(c,A,b,Aeq,beq,tol,maxit)

solveLP : {} -> Ndarray
solveLP = Native.Numeric.solveLP

solveQP(Dmat, dvec, Amat, bvec, meq, factorized) 

solveQP : {} -> Ndarray
solveQP = Native.Numeric.solveQP

-}

{-| Boolean matrix operations 
    All resulting matrices are either Boolean matrices or Booleans -}

all : Ndarray -> Bool
all = Native.Numeric.all

and : Ndarray -> Ndarray -> Ndarray
and = Native.Numeric.and

any : Ndarray -> Bool
any = Native.Numeric.any

eq : Ndarray -> Ndarray -> Ndarray
eq = Native.Numeric.eq

geq : Ndarray -> Ndarray -> Ndarray
geq = Native.Numeric.geq

gt : Ndarray -> Ndarray -> Ndarray
gt = Native.Numeric.gt

isFinite : Ndarray -> Ndarray
isFinite = Native.Numeric.isFinite

isNaN : Ndarray -> Ndarray
isNaN = Native.Numeric.isNaN

leq : Ndarray -> Ndarray -> Ndarray
leq = Native.Numeric.leq

lt : Ndarray -> Ndarray -> Ndarray
lt = Native.Numeric.lt

neq : Ndarray -> Ndarray -> Ndarray
neq = Native.Numeric.neq

not : Ndarray -> Ndarray
not = Native.Numeric.not

or : Ndarray -> Ndarray -> Ndarray
or = Native.Numeric.or

same : Ndarray -> Ndarray -> Bool
same = Native.Numeric.same

xor : Ndarray -> Ndarray -> Ndarray
xor = Native.Numeric.xor

{-| Elm representation -}

toList : Ndarray -> List a
toList = Native.Numeric.toList

--toCSV : Ndarray -> IO.IO
--toCSV = Native.Numeric.toCSV

print : Ndarray -> String
print = Native.Numeric.print
