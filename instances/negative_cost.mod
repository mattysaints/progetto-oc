
    set I;
param n := card(I);

set SS := 0 .. (2**n - 1);

set POW {k in SS} := {i in I: (k div 2**(i-1)) mod 2 = 1};

set INFINITE := {(1, 3), (1, 5), (1, 6), (1, 8), (1, 9), (2, 3), (2, 5), (2, 7), (2, 8), (3, 5), (3, 8), (3, 9), (4, 6), (4, 7), (4, 9), (5, 6), (5, 8), (6, 7), (7, 9)};

set LINKS := {i in I, j in I: i < j};

param cost {LINKS};
var x {LINKS} binary;

minimize TotCost: sum {(i,j) in LINKS} cost[i,j] * x[i,j];

subj to Tour {i in I}: 
   sum {(i,j) in LINKS} x[i,j] + sum {(j,i) in LINKS} x[j,i] = 2;

subj to SubtourElim {k in SS diff {0,2**n-1}}:
   sum {i in POW[k], j in I diff POW[k]: (i,j) in LINKS} x[i,j] +
   sum {i in POW[k], j in I diff POW[k]: (j,i) in LINKS} x[j,i] >= 2;
   
subj to Inf{(i,j) in INFINITE}: x[i,j] = 0;

solve;

printf "------------------------------------------------------\n";
printf{i in I, j in I: j>i and x[i,j] == 1} "(%d, %d)\n", i-1, j-1;
printf "Cost: %d\n", sum{i in I, j in I: j > i} x[i,j]*cost[i,j];
printf "------------------------------------------------------\n";

data;

set I := 1 2 3 4 5 6 7 8 9;

param cost: 1 2 3 4 5 6 7 8 9 :=
	1 . 10.0 0 5.0 0 0 14.0 0 0
	2 . . 0 7.0 0 2.0 0 0 32.0
	3 . . . 3.0 0 12.0 1.0 0 0
	4 . . . . 10.0 0 0 -11.0 0
	5 . . . . . 0 1.0 0 9.0
	6 . . . . . . 0 -1.0 -4.0
	7 . . . . . . . 1.0 0
	8 . . . . . . . . 5.0
	9 . . . . . . . . .;

end;
