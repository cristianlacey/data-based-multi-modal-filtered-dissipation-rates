��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
\
	LeakyRelu
features"T
activations"T"
alphafloat%��L>"
Ttype0:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
:
Maximum
x"T
y"T
z"T"
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
x"T
y"T"
Ttype:

2
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.12unknown8��
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
�
dense_0_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*%
shared_namedense_0_nn_36/kernel
}
(dense_0_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_0_nn_36/kernel*
_output_shapes

:$*
dtype0
|
dense_0_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_0_nn_36/bias
u
&dense_0_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_0_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_1_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_1_nn_36/kernel
}
(dense_1_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_1_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_1_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_1_nn_36/bias
u
&dense_1_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_1_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_2_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_2_nn_36/kernel
}
(dense_2_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_2_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_2_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_2_nn_36/bias
u
&dense_2_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_2_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_3_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_3_nn_36/kernel
}
(dense_3_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_3_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_3_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_3_nn_36/bias
u
&dense_3_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_3_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_4_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_4_nn_36/kernel
}
(dense_4_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_4_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_4_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_4_nn_36/bias
u
&dense_4_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_4_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_5_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_5_nn_36/kernel
}
(dense_5_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_5_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_5_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_5_nn_36/bias
u
&dense_5_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_5_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_6_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_6_nn_36/kernel
}
(dense_6_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_6_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_6_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_6_nn_36/bias
u
&dense_6_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_6_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_7_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*%
shared_namedense_7_nn_36/kernel
}
(dense_7_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_7_nn_36/kernel*
_output_shapes

:$$*
dtype0
|
dense_7_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:$*#
shared_namedense_7_nn_36/bias
u
&dense_7_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_7_nn_36/bias*
_output_shapes
:$*
dtype0
�
dense_out_nl_8_nn_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*,
shared_namedense_out_nl_8_nn_36/kernel
�
/dense_out_nl_8_nn_36/kernel/Read/ReadVariableOpReadVariableOpdense_out_nl_8_nn_36/kernel*
_output_shapes

:$*
dtype0
�
dense_out_nl_8_nn_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_namedense_out_nl_8_nn_36/bias
�
-dense_out_nl_8_nn_36/bias/Read/ReadVariableOpReadVariableOpdense_out_nl_8_nn_36/bias*
_output_shapes
:*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
�
Adam/dense_0_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*,
shared_nameAdam/dense_0_nn_36/kernel/m
�
/Adam/dense_0_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_0_nn_36/kernel/m*
_output_shapes

:$*
dtype0
�
Adam/dense_0_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_0_nn_36/bias/m
�
-Adam/dense_0_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_0_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_1_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_1_nn_36/kernel/m
�
/Adam/dense_1_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_1_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_1_nn_36/bias/m
�
-Adam/dense_1_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_2_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_2_nn_36/kernel/m
�
/Adam/dense_2_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_2_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_2_nn_36/bias/m
�
-Adam/dense_2_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_3_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_3_nn_36/kernel/m
�
/Adam/dense_3_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_3_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_3_nn_36/bias/m
�
-Adam/dense_3_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_4_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_4_nn_36/kernel/m
�
/Adam/dense_4_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_4_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_4_nn_36/bias/m
�
-Adam/dense_4_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_5_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_5_nn_36/kernel/m
�
/Adam/dense_5_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_5_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_5_nn_36/bias/m
�
-Adam/dense_5_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_6_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_6_nn_36/kernel/m
�
/Adam/dense_6_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_6_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_6_nn_36/bias/m
�
-Adam/dense_6_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
Adam/dense_7_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_7_nn_36/kernel/m
�
/Adam/dense_7_nn_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7_nn_36/kernel/m*
_output_shapes

:$$*
dtype0
�
Adam/dense_7_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_7_nn_36/bias/m
�
-Adam/dense_7_nn_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7_nn_36/bias/m*
_output_shapes
:$*
dtype0
�
"Adam/dense_out_nl_8_nn_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*3
shared_name$"Adam/dense_out_nl_8_nn_36/kernel/m
�
6Adam/dense_out_nl_8_nn_36/kernel/m/Read/ReadVariableOpReadVariableOp"Adam/dense_out_nl_8_nn_36/kernel/m*
_output_shapes

:$*
dtype0
�
 Adam/dense_out_nl_8_nn_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/dense_out_nl_8_nn_36/bias/m
�
4Adam/dense_out_nl_8_nn_36/bias/m/Read/ReadVariableOpReadVariableOp Adam/dense_out_nl_8_nn_36/bias/m*
_output_shapes
:*
dtype0
�
Adam/dense_0_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*,
shared_nameAdam/dense_0_nn_36/kernel/v
�
/Adam/dense_0_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_0_nn_36/kernel/v*
_output_shapes

:$*
dtype0
�
Adam/dense_0_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_0_nn_36/bias/v
�
-Adam/dense_0_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_0_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_1_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_1_nn_36/kernel/v
�
/Adam/dense_1_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_1_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_1_nn_36/bias/v
�
-Adam/dense_1_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_2_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_2_nn_36/kernel/v
�
/Adam/dense_2_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_2_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_2_nn_36/bias/v
�
-Adam/dense_2_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_3_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_3_nn_36/kernel/v
�
/Adam/dense_3_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_3_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_3_nn_36/bias/v
�
-Adam/dense_3_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_4_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_4_nn_36/kernel/v
�
/Adam/dense_4_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_4_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_4_nn_36/bias/v
�
-Adam/dense_4_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_5_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_5_nn_36/kernel/v
�
/Adam/dense_5_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_5_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_5_nn_36/bias/v
�
-Adam/dense_5_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_6_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_6_nn_36/kernel/v
�
/Adam/dense_6_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_6_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_6_nn_36/bias/v
�
-Adam/dense_6_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
Adam/dense_7_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$$*,
shared_nameAdam/dense_7_nn_36/kernel/v
�
/Adam/dense_7_nn_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7_nn_36/kernel/v*
_output_shapes

:$$*
dtype0
�
Adam/dense_7_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:$**
shared_nameAdam/dense_7_nn_36/bias/v
�
-Adam/dense_7_nn_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7_nn_36/bias/v*
_output_shapes
:$*
dtype0
�
"Adam/dense_out_nl_8_nn_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:$*3
shared_name$"Adam/dense_out_nl_8_nn_36/kernel/v
�
6Adam/dense_out_nl_8_nn_36/kernel/v/Read/ReadVariableOpReadVariableOp"Adam/dense_out_nl_8_nn_36/kernel/v*
_output_shapes

:$*
dtype0
�
 Adam/dense_out_nl_8_nn_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*1
shared_name" Adam/dense_out_nl_8_nn_36/bias/v
�
4Adam/dense_out_nl_8_nn_36/bias/v/Read/ReadVariableOpReadVariableOp Adam/dense_out_nl_8_nn_36/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
�q
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�q
value�qB�q B�q
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
]
state_variables
_broadcast_shape
mean
variance
	count
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
R
%regularization_losses
&	variables
'trainable_variables
(	keras_api
h

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
R
/regularization_losses
0	variables
1trainable_variables
2	keras_api
h

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
R
9regularization_losses
:	variables
;trainable_variables
<	keras_api
h

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
R
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
h

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
R
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
h

Qkernel
Rbias
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
R
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
R
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
h

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
R
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
h

okernel
pbias
qregularization_losses
r	variables
strainable_variables
t	keras_api
�
uiter

vbeta_1

wbeta_2
	xdecay
ylearning_ratem� m�)m�*m�3m�4m�=m�>m�Gm�Hm�Qm�Rm�[m�\m�em�fm�om�pm�v� v�)v�*v�3v�4v�=v�>v�Gv�Hv�Qv�Rv�[v�\v�ev�fv�ov�pv�
 
�
0
1
2
3
 4
)5
*6
37
48
=9
>10
G11
H12
Q13
R14
[15
\16
e17
f18
o19
p20
�
0
 1
)2
*3
34
45
=6
>7
G8
H9
Q10
R11
[12
\13
e14
f15
o16
p17
�
regularization_losses
zmetrics
{layer_metrics
|non_trainable_variables
	variables

}layers
trainable_variables
~layer_regularization_losses
 
#
mean
variance
	count
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 
`^
VARIABLE_VALUEdense_0_nn_36/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_0_nn_36/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
�
!regularization_losses
metrics
�layer_metrics
�non_trainable_variables
"	variables
�layers
#trainable_variables
 �layer_regularization_losses
 
 
 
�
%regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
&	variables
�layers
'trainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_1_nn_36/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_1_nn_36/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

)0
*1

)0
*1
�
+regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
,	variables
�layers
-trainable_variables
 �layer_regularization_losses
 
 
 
�
/regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
0	variables
�layers
1trainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_2_nn_36/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_2_nn_36/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

30
41

30
41
�
5regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
6	variables
�layers
7trainable_variables
 �layer_regularization_losses
 
 
 
�
9regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
:	variables
�layers
;trainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_3_nn_36/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_3_nn_36/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

=0
>1

=0
>1
�
?regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
@	variables
�layers
Atrainable_variables
 �layer_regularization_losses
 
 
 
�
Cregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
D	variables
�layers
Etrainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_4_nn_36/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_4_nn_36/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE
 

G0
H1

G0
H1
�
Iregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
J	variables
�layers
Ktrainable_variables
 �layer_regularization_losses
 
 
 
�
Mregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
N	variables
�layers
Otrainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_5_nn_36/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_5_nn_36/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

Q0
R1

Q0
R1
�
Sregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
T	variables
�layers
Utrainable_variables
 �layer_regularization_losses
 
 
 
�
Wregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
X	variables
�layers
Ytrainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_6_nn_36/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_6_nn_36/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
�
]regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
^	variables
�layers
_trainable_variables
 �layer_regularization_losses
 
 
 
�
aregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
b	variables
�layers
ctrainable_variables
 �layer_regularization_losses
`^
VARIABLE_VALUEdense_7_nn_36/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
\Z
VARIABLE_VALUEdense_7_nn_36/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

e0
f1

e0
f1
�
gregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
h	variables
�layers
itrainable_variables
 �layer_regularization_losses
 
 
 
�
kregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
l	variables
�layers
mtrainable_variables
 �layer_regularization_losses
ge
VARIABLE_VALUEdense_out_nl_8_nn_36/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEdense_out_nl_8_nn_36/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE
 

o0
p1

o0
p1
�
qregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
r	variables
�layers
strainable_variables
 �layer_regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE

�0
 

0
1
2
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
8

�total

�count
�	variables
�	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

�0
�1

�	variables
��
VARIABLE_VALUEAdam/dense_0_nn_36/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_0_nn_36/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_1_nn_36/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_1_nn_36/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_2_nn_36/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2_nn_36/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_3_nn_36/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_3_nn_36/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_4_nn_36/kernel/mRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_4_nn_36/bias/mPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_5_nn_36/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_5_nn_36/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_6_nn_36/kernel/mRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_6_nn_36/bias/mPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_7_nn_36/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_7_nn_36/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/dense_out_nl_8_nn_36/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/dense_out_nl_8_nn_36/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_0_nn_36/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_0_nn_36/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_1_nn_36/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_1_nn_36/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_2_nn_36/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_2_nn_36/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_3_nn_36/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_3_nn_36/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_4_nn_36/kernel/vRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_4_nn_36/bias/vPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_5_nn_36/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_5_nn_36/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_6_nn_36/kernel/vRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_6_nn_36/bias/vPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUEAdam/dense_7_nn_36/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/dense_7_nn_36/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE"Adam/dense_out_nl_8_nn_36/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
��
VARIABLE_VALUE Adam/dense_out_nl_8_nn_36/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
�
#serving_default_normalization_inputPlaceholder*0
_output_shapes
:������������������*
dtype0*%
shape:������������������
�
StatefulPartitionedCallStatefulPartitionedCall#serving_default_normalization_inputmeanvariancedense_0_nn_36/kerneldense_0_nn_36/biasdense_1_nn_36/kerneldense_1_nn_36/biasdense_2_nn_36/kerneldense_2_nn_36/biasdense_3_nn_36/kerneldense_3_nn_36/biasdense_4_nn_36/kerneldense_4_nn_36/biasdense_5_nn_36/kerneldense_5_nn_36/biasdense_6_nn_36/kerneldense_6_nn_36/biasdense_7_nn_36/kerneldense_7_nn_36/biasdense_out_nl_8_nn_36/kerneldense_out_nl_8_nn_36/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *.
f)R'
%__inference_signature_wrapper_2193474
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp(dense_0_nn_36/kernel/Read/ReadVariableOp&dense_0_nn_36/bias/Read/ReadVariableOp(dense_1_nn_36/kernel/Read/ReadVariableOp&dense_1_nn_36/bias/Read/ReadVariableOp(dense_2_nn_36/kernel/Read/ReadVariableOp&dense_2_nn_36/bias/Read/ReadVariableOp(dense_3_nn_36/kernel/Read/ReadVariableOp&dense_3_nn_36/bias/Read/ReadVariableOp(dense_4_nn_36/kernel/Read/ReadVariableOp&dense_4_nn_36/bias/Read/ReadVariableOp(dense_5_nn_36/kernel/Read/ReadVariableOp&dense_5_nn_36/bias/Read/ReadVariableOp(dense_6_nn_36/kernel/Read/ReadVariableOp&dense_6_nn_36/bias/Read/ReadVariableOp(dense_7_nn_36/kernel/Read/ReadVariableOp&dense_7_nn_36/bias/Read/ReadVariableOp/dense_out_nl_8_nn_36/kernel/Read/ReadVariableOp-dense_out_nl_8_nn_36/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOp/Adam/dense_0_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_0_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_1_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_1_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_2_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_2_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_3_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_3_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_4_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_4_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_5_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_5_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_6_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_6_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_7_nn_36/kernel/m/Read/ReadVariableOp-Adam/dense_7_nn_36/bias/m/Read/ReadVariableOp6Adam/dense_out_nl_8_nn_36/kernel/m/Read/ReadVariableOp4Adam/dense_out_nl_8_nn_36/bias/m/Read/ReadVariableOp/Adam/dense_0_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_0_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_1_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_1_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_2_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_2_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_3_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_3_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_4_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_4_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_5_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_5_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_6_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_6_nn_36/bias/v/Read/ReadVariableOp/Adam/dense_7_nn_36/kernel/v/Read/ReadVariableOp-Adam/dense_7_nn_36/bias/v/Read/ReadVariableOp6Adam/dense_out_nl_8_nn_36/kernel/v/Read/ReadVariableOp4Adam/dense_out_nl_8_nn_36/bias/v/Read/ReadVariableOpConst*M
TinF
D2B		*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *)
f$R"
 __inference__traced_save_2194188
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountdense_0_nn_36/kerneldense_0_nn_36/biasdense_1_nn_36/kerneldense_1_nn_36/biasdense_2_nn_36/kerneldense_2_nn_36/biasdense_3_nn_36/kerneldense_3_nn_36/biasdense_4_nn_36/kerneldense_4_nn_36/biasdense_5_nn_36/kerneldense_5_nn_36/biasdense_6_nn_36/kerneldense_6_nn_36/biasdense_7_nn_36/kerneldense_7_nn_36/biasdense_out_nl_8_nn_36/kerneldense_out_nl_8_nn_36/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcount_1Adam/dense_0_nn_36/kernel/mAdam/dense_0_nn_36/bias/mAdam/dense_1_nn_36/kernel/mAdam/dense_1_nn_36/bias/mAdam/dense_2_nn_36/kernel/mAdam/dense_2_nn_36/bias/mAdam/dense_3_nn_36/kernel/mAdam/dense_3_nn_36/bias/mAdam/dense_4_nn_36/kernel/mAdam/dense_4_nn_36/bias/mAdam/dense_5_nn_36/kernel/mAdam/dense_5_nn_36/bias/mAdam/dense_6_nn_36/kernel/mAdam/dense_6_nn_36/bias/mAdam/dense_7_nn_36/kernel/mAdam/dense_7_nn_36/bias/m"Adam/dense_out_nl_8_nn_36/kernel/m Adam/dense_out_nl_8_nn_36/bias/mAdam/dense_0_nn_36/kernel/vAdam/dense_0_nn_36/bias/vAdam/dense_1_nn_36/kernel/vAdam/dense_1_nn_36/bias/vAdam/dense_2_nn_36/kernel/vAdam/dense_2_nn_36/bias/vAdam/dense_3_nn_36/kernel/vAdam/dense_3_nn_36/bias/vAdam/dense_4_nn_36/kernel/vAdam/dense_4_nn_36/bias/vAdam/dense_5_nn_36/kernel/vAdam/dense_5_nn_36/bias/vAdam/dense_6_nn_36/kernel/vAdam/dense_6_nn_36/bias/vAdam/dense_7_nn_36/kernel/vAdam/dense_7_nn_36/bias/v"Adam/dense_out_nl_8_nn_36/kernel/v Adam/dense_out_nl_8_nn_36/bias/v*L
TinE
C2A*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *,
f'R%
#__inference__traced_restore_2194390��
�`
�	
G__inference_sequential_layer_call_and_return_conditional_losses_2193261

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_0_nn_36_2193207
dense_0_nn_36_2193209
dense_1_nn_36_2193213
dense_1_nn_36_2193215
dense_2_nn_36_2193219
dense_2_nn_36_2193221
dense_3_nn_36_2193225
dense_3_nn_36_2193227
dense_4_nn_36_2193231
dense_4_nn_36_2193233
dense_5_nn_36_2193237
dense_5_nn_36_2193239
dense_6_nn_36_2193243
dense_6_nn_36_2193245
dense_7_nn_36_2193249
dense_7_nn_36_2193251 
dense_out_nl_8_nn_36_2193255 
dense_out_nl_8_nn_36_2193257
identity��%dense_0_nn_36/StatefulPartitionedCall�%dense_1_nn_36/StatefulPartitionedCall�%dense_2_nn_36/StatefulPartitionedCall�%dense_3_nn_36/StatefulPartitionedCall�%dense_4_nn_36/StatefulPartitionedCall�%dense_5_nn_36/StatefulPartitionedCall�%dense_6_nn_36/StatefulPartitionedCall�%dense_7_nn_36/StatefulPartitionedCall�,dense_out_nl_8_nn_36/StatefulPartitionedCall�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
%dense_0_nn_36/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_0_nn_36_2193207dense_0_nn_36_2193209*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_21927892'
%dense_0_nn_36/StatefulPartitionedCall�
#leaky_re_lu_0_nn_36/PartitionedCallPartitionedCall.dense_0_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_21928102%
#leaky_re_lu_0_nn_36/PartitionedCall�
%dense_1_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_0_nn_36/PartitionedCall:output:0dense_1_nn_36_2193213dense_1_nn_36_2193215*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_21928282'
%dense_1_nn_36/StatefulPartitionedCall�
#leaky_re_lu_1_nn_36/PartitionedCallPartitionedCall.dense_1_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_21928492%
#leaky_re_lu_1_nn_36/PartitionedCall�
%dense_2_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_1_nn_36/PartitionedCall:output:0dense_2_nn_36_2193219dense_2_nn_36_2193221*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_21928672'
%dense_2_nn_36/StatefulPartitionedCall�
#leaky_re_lu_2_nn_36/PartitionedCallPartitionedCall.dense_2_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_21928882%
#leaky_re_lu_2_nn_36/PartitionedCall�
%dense_3_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_2_nn_36/PartitionedCall:output:0dense_3_nn_36_2193225dense_3_nn_36_2193227*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_21929062'
%dense_3_nn_36/StatefulPartitionedCall�
#leaky_re_lu_3_nn_36/PartitionedCallPartitionedCall.dense_3_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_21929272%
#leaky_re_lu_3_nn_36/PartitionedCall�
%dense_4_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_3_nn_36/PartitionedCall:output:0dense_4_nn_36_2193231dense_4_nn_36_2193233*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_21929452'
%dense_4_nn_36/StatefulPartitionedCall�
#leaky_re_lu_4_nn_36/PartitionedCallPartitionedCall.dense_4_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_21929662%
#leaky_re_lu_4_nn_36/PartitionedCall�
%dense_5_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_4_nn_36/PartitionedCall:output:0dense_5_nn_36_2193237dense_5_nn_36_2193239*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_21929842'
%dense_5_nn_36/StatefulPartitionedCall�
#leaky_re_lu_5_nn_36/PartitionedCallPartitionedCall.dense_5_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_21930052%
#leaky_re_lu_5_nn_36/PartitionedCall�
%dense_6_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_5_nn_36/PartitionedCall:output:0dense_6_nn_36_2193243dense_6_nn_36_2193245*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_21930232'
%dense_6_nn_36/StatefulPartitionedCall�
#leaky_re_lu_6_nn_36/PartitionedCallPartitionedCall.dense_6_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_21930442%
#leaky_re_lu_6_nn_36/PartitionedCall�
%dense_7_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_6_nn_36/PartitionedCall:output:0dense_7_nn_36_2193249dense_7_nn_36_2193251*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_21930622'
%dense_7_nn_36/StatefulPartitionedCall�
#leaky_re_lu_7_nn_36/PartitionedCallPartitionedCall.dense_7_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_21930832%
#leaky_re_lu_7_nn_36/PartitionedCall�
,dense_out_nl_8_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_7_nn_36/PartitionedCall:output:0dense_out_nl_8_nn_36_2193255dense_out_nl_8_nn_36_2193257*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_21931012.
,dense_out_nl_8_nn_36/StatefulPartitionedCall�
IdentityIdentity5dense_out_nl_8_nn_36/StatefulPartitionedCall:output:0&^dense_0_nn_36/StatefulPartitionedCall&^dense_1_nn_36/StatefulPartitionedCall&^dense_2_nn_36/StatefulPartitionedCall&^dense_3_nn_36/StatefulPartitionedCall&^dense_4_nn_36/StatefulPartitionedCall&^dense_5_nn_36/StatefulPartitionedCall&^dense_6_nn_36/StatefulPartitionedCall&^dense_7_nn_36/StatefulPartitionedCall-^dense_out_nl_8_nn_36/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2N
%dense_0_nn_36/StatefulPartitionedCall%dense_0_nn_36/StatefulPartitionedCall2N
%dense_1_nn_36/StatefulPartitionedCall%dense_1_nn_36/StatefulPartitionedCall2N
%dense_2_nn_36/StatefulPartitionedCall%dense_2_nn_36/StatefulPartitionedCall2N
%dense_3_nn_36/StatefulPartitionedCall%dense_3_nn_36/StatefulPartitionedCall2N
%dense_4_nn_36/StatefulPartitionedCall%dense_4_nn_36/StatefulPartitionedCall2N
%dense_5_nn_36/StatefulPartitionedCall%dense_5_nn_36/StatefulPartitionedCall2N
%dense_6_nn_36/StatefulPartitionedCall%dense_6_nn_36/StatefulPartitionedCall2N
%dense_7_nn_36/StatefulPartitionedCall%dense_7_nn_36/StatefulPartitionedCall2\
,dense_out_nl_8_nn_36/StatefulPartitionedCall,dense_out_nl_8_nn_36/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�	
�
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_2193761

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_2192927

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_2192945

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_2193044

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_2193964

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_1_nn_36_layer_call_fn_2193780

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_21928492
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_2192867

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_4_nn_36_layer_call_fn_2193857

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_21929452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
%__inference_signature_wrapper_2193474
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *+
f&R$
"__inference__wrapped_model_21927622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
Q
5__inference_leaky_re_lu_0_nn_36_layer_call_fn_2193751

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_21928102
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_2193891

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
׊
�
"__inference__wrapped_model_2192762
normalization_input<
8sequential_normalization_reshape_readvariableop_resource>
:sequential_normalization_reshape_1_readvariableop_resource;
7sequential_dense_0_nn_36_matmul_readvariableop_resource<
8sequential_dense_0_nn_36_biasadd_readvariableop_resource;
7sequential_dense_1_nn_36_matmul_readvariableop_resource<
8sequential_dense_1_nn_36_biasadd_readvariableop_resource;
7sequential_dense_2_nn_36_matmul_readvariableop_resource<
8sequential_dense_2_nn_36_biasadd_readvariableop_resource;
7sequential_dense_3_nn_36_matmul_readvariableop_resource<
8sequential_dense_3_nn_36_biasadd_readvariableop_resource;
7sequential_dense_4_nn_36_matmul_readvariableop_resource<
8sequential_dense_4_nn_36_biasadd_readvariableop_resource;
7sequential_dense_5_nn_36_matmul_readvariableop_resource<
8sequential_dense_5_nn_36_biasadd_readvariableop_resource;
7sequential_dense_6_nn_36_matmul_readvariableop_resource<
8sequential_dense_6_nn_36_biasadd_readvariableop_resource;
7sequential_dense_7_nn_36_matmul_readvariableop_resource<
8sequential_dense_7_nn_36_biasadd_readvariableop_resourceB
>sequential_dense_out_nl_8_nn_36_matmul_readvariableop_resourceC
?sequential_dense_out_nl_8_nn_36_biasadd_readvariableop_resource
identity��/sequential/dense_0_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_0_nn_36/MatMul/ReadVariableOp�/sequential/dense_1_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_1_nn_36/MatMul/ReadVariableOp�/sequential/dense_2_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_2_nn_36/MatMul/ReadVariableOp�/sequential/dense_3_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_3_nn_36/MatMul/ReadVariableOp�/sequential/dense_4_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_4_nn_36/MatMul/ReadVariableOp�/sequential/dense_5_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_5_nn_36/MatMul/ReadVariableOp�/sequential/dense_6_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_6_nn_36/MatMul/ReadVariableOp�/sequential/dense_7_nn_36/BiasAdd/ReadVariableOp�.sequential/dense_7_nn_36/MatMul/ReadVariableOp�6sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�5sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp�/sequential/normalization/Reshape/ReadVariableOp�1sequential/normalization/Reshape_1/ReadVariableOp�
/sequential/normalization/Reshape/ReadVariableOpReadVariableOp8sequential_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype021
/sequential/normalization/Reshape/ReadVariableOp�
&sequential/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2(
&sequential/normalization/Reshape/shape�
 sequential/normalization/ReshapeReshape7sequential/normalization/Reshape/ReadVariableOp:value:0/sequential/normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2"
 sequential/normalization/Reshape�
1sequential/normalization/Reshape_1/ReadVariableOpReadVariableOp:sequential_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype023
1sequential/normalization/Reshape_1/ReadVariableOp�
(sequential/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2*
(sequential/normalization/Reshape_1/shape�
"sequential/normalization/Reshape_1Reshape9sequential/normalization/Reshape_1/ReadVariableOp:value:01sequential/normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2$
"sequential/normalization/Reshape_1�
sequential/normalization/subSubnormalization_input)sequential/normalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
sequential/normalization/sub�
sequential/normalization/SqrtSqrt+sequential/normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
sequential/normalization/Sqrt�
"sequential/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32$
"sequential/normalization/Maximum/y�
 sequential/normalization/MaximumMaximum!sequential/normalization/Sqrt:y:0+sequential/normalization/Maximum/y:output:0*
T0*
_output_shapes

:2"
 sequential/normalization/Maximum�
 sequential/normalization/truedivRealDiv sequential/normalization/sub:z:0$sequential/normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2"
 sequential/normalization/truediv�
.sequential/dense_0_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_0_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype020
.sequential/dense_0_nn_36/MatMul/ReadVariableOp�
sequential/dense_0_nn_36/MatMulMatMul$sequential/normalization/truediv:z:06sequential/dense_0_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_0_nn_36/MatMul�
/sequential/dense_0_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_0_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_0_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_0_nn_36/BiasAddBiasAdd)sequential/dense_0_nn_36/MatMul:product:07sequential/dense_0_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_0_nn_36/BiasAdd�
(sequential/leaky_re_lu_0_nn_36/LeakyRelu	LeakyRelu)sequential/dense_0_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_0_nn_36/LeakyRelu�
.sequential/dense_1_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_1_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_1_nn_36/MatMul/ReadVariableOp�
sequential/dense_1_nn_36/MatMulMatMul6sequential/leaky_re_lu_0_nn_36/LeakyRelu:activations:06sequential/dense_1_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_1_nn_36/MatMul�
/sequential/dense_1_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_1_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_1_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_1_nn_36/BiasAddBiasAdd)sequential/dense_1_nn_36/MatMul:product:07sequential/dense_1_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_1_nn_36/BiasAdd�
(sequential/leaky_re_lu_1_nn_36/LeakyRelu	LeakyRelu)sequential/dense_1_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_1_nn_36/LeakyRelu�
.sequential/dense_2_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_2_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_2_nn_36/MatMul/ReadVariableOp�
sequential/dense_2_nn_36/MatMulMatMul6sequential/leaky_re_lu_1_nn_36/LeakyRelu:activations:06sequential/dense_2_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_2_nn_36/MatMul�
/sequential/dense_2_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_2_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_2_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_2_nn_36/BiasAddBiasAdd)sequential/dense_2_nn_36/MatMul:product:07sequential/dense_2_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_2_nn_36/BiasAdd�
(sequential/leaky_re_lu_2_nn_36/LeakyRelu	LeakyRelu)sequential/dense_2_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_2_nn_36/LeakyRelu�
.sequential/dense_3_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_3_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_3_nn_36/MatMul/ReadVariableOp�
sequential/dense_3_nn_36/MatMulMatMul6sequential/leaky_re_lu_2_nn_36/LeakyRelu:activations:06sequential/dense_3_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_3_nn_36/MatMul�
/sequential/dense_3_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_3_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_3_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_3_nn_36/BiasAddBiasAdd)sequential/dense_3_nn_36/MatMul:product:07sequential/dense_3_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_3_nn_36/BiasAdd�
(sequential/leaky_re_lu_3_nn_36/LeakyRelu	LeakyRelu)sequential/dense_3_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_3_nn_36/LeakyRelu�
.sequential/dense_4_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_4_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_4_nn_36/MatMul/ReadVariableOp�
sequential/dense_4_nn_36/MatMulMatMul6sequential/leaky_re_lu_3_nn_36/LeakyRelu:activations:06sequential/dense_4_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_4_nn_36/MatMul�
/sequential/dense_4_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_4_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_4_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_4_nn_36/BiasAddBiasAdd)sequential/dense_4_nn_36/MatMul:product:07sequential/dense_4_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_4_nn_36/BiasAdd�
(sequential/leaky_re_lu_4_nn_36/LeakyRelu	LeakyRelu)sequential/dense_4_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_4_nn_36/LeakyRelu�
.sequential/dense_5_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_5_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_5_nn_36/MatMul/ReadVariableOp�
sequential/dense_5_nn_36/MatMulMatMul6sequential/leaky_re_lu_4_nn_36/LeakyRelu:activations:06sequential/dense_5_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_5_nn_36/MatMul�
/sequential/dense_5_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_5_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_5_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_5_nn_36/BiasAddBiasAdd)sequential/dense_5_nn_36/MatMul:product:07sequential/dense_5_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_5_nn_36/BiasAdd�
(sequential/leaky_re_lu_5_nn_36/LeakyRelu	LeakyRelu)sequential/dense_5_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_5_nn_36/LeakyRelu�
.sequential/dense_6_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_6_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_6_nn_36/MatMul/ReadVariableOp�
sequential/dense_6_nn_36/MatMulMatMul6sequential/leaky_re_lu_5_nn_36/LeakyRelu:activations:06sequential/dense_6_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_6_nn_36/MatMul�
/sequential/dense_6_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_6_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_6_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_6_nn_36/BiasAddBiasAdd)sequential/dense_6_nn_36/MatMul:product:07sequential/dense_6_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_6_nn_36/BiasAdd�
(sequential/leaky_re_lu_6_nn_36/LeakyRelu	LeakyRelu)sequential/dense_6_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_6_nn_36/LeakyRelu�
.sequential/dense_7_nn_36/MatMul/ReadVariableOpReadVariableOp7sequential_dense_7_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype020
.sequential/dense_7_nn_36/MatMul/ReadVariableOp�
sequential/dense_7_nn_36/MatMulMatMul6sequential/leaky_re_lu_6_nn_36/LeakyRelu:activations:06sequential/dense_7_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2!
sequential/dense_7_nn_36/MatMul�
/sequential/dense_7_nn_36/BiasAdd/ReadVariableOpReadVariableOp8sequential_dense_7_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype021
/sequential/dense_7_nn_36/BiasAdd/ReadVariableOp�
 sequential/dense_7_nn_36/BiasAddBiasAdd)sequential/dense_7_nn_36/MatMul:product:07sequential/dense_7_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2"
 sequential/dense_7_nn_36/BiasAdd�
(sequential/leaky_re_lu_7_nn_36/LeakyRelu	LeakyRelu)sequential/dense_7_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2*
(sequential/leaky_re_lu_7_nn_36/LeakyRelu�
5sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOpReadVariableOp>sequential_dense_out_nl_8_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype027
5sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp�
&sequential/dense_out_nl_8_nn_36/MatMulMatMul6sequential/leaky_re_lu_7_nn_36/LeakyRelu:activations:0=sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2(
&sequential/dense_out_nl_8_nn_36/MatMul�
6sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOpReadVariableOp?sequential_dense_out_nl_8_nn_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype028
6sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�
'sequential/dense_out_nl_8_nn_36/BiasAddBiasAdd0sequential/dense_out_nl_8_nn_36/MatMul:product:0>sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2)
'sequential/dense_out_nl_8_nn_36/BiasAdd�
IdentityIdentity0sequential/dense_out_nl_8_nn_36/BiasAdd:output:00^sequential/dense_0_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_0_nn_36/MatMul/ReadVariableOp0^sequential/dense_1_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_1_nn_36/MatMul/ReadVariableOp0^sequential/dense_2_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_2_nn_36/MatMul/ReadVariableOp0^sequential/dense_3_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_3_nn_36/MatMul/ReadVariableOp0^sequential/dense_4_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_4_nn_36/MatMul/ReadVariableOp0^sequential/dense_5_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_5_nn_36/MatMul/ReadVariableOp0^sequential/dense_6_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_6_nn_36/MatMul/ReadVariableOp0^sequential/dense_7_nn_36/BiasAdd/ReadVariableOp/^sequential/dense_7_nn_36/MatMul/ReadVariableOp7^sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp6^sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp0^sequential/normalization/Reshape/ReadVariableOp2^sequential/normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2b
/sequential/dense_0_nn_36/BiasAdd/ReadVariableOp/sequential/dense_0_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_0_nn_36/MatMul/ReadVariableOp.sequential/dense_0_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_1_nn_36/BiasAdd/ReadVariableOp/sequential/dense_1_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_1_nn_36/MatMul/ReadVariableOp.sequential/dense_1_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_2_nn_36/BiasAdd/ReadVariableOp/sequential/dense_2_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_2_nn_36/MatMul/ReadVariableOp.sequential/dense_2_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_3_nn_36/BiasAdd/ReadVariableOp/sequential/dense_3_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_3_nn_36/MatMul/ReadVariableOp.sequential/dense_3_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_4_nn_36/BiasAdd/ReadVariableOp/sequential/dense_4_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_4_nn_36/MatMul/ReadVariableOp.sequential/dense_4_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_5_nn_36/BiasAdd/ReadVariableOp/sequential/dense_5_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_5_nn_36/MatMul/ReadVariableOp.sequential/dense_5_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_6_nn_36/BiasAdd/ReadVariableOp/sequential/dense_6_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_6_nn_36/MatMul/ReadVariableOp.sequential/dense_6_nn_36/MatMul/ReadVariableOp2b
/sequential/dense_7_nn_36/BiasAdd/ReadVariableOp/sequential/dense_7_nn_36/BiasAdd/ReadVariableOp2`
.sequential/dense_7_nn_36/MatMul/ReadVariableOp.sequential/dense_7_nn_36/MatMul/ReadVariableOp2p
6sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp6sequential/dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp2n
5sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp5sequential/dense_out_nl_8_nn_36/MatMul/ReadVariableOp2b
/sequential/normalization/Reshape/ReadVariableOp/sequential/normalization/Reshape/ReadVariableOp2f
1sequential/normalization/Reshape_1/ReadVariableOp1sequential/normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�	
�
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_2193790

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_2193949

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
��
�
 __inference__traced_save_2194188
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	3
/savev2_dense_0_nn_36_kernel_read_readvariableop1
-savev2_dense_0_nn_36_bias_read_readvariableop3
/savev2_dense_1_nn_36_kernel_read_readvariableop1
-savev2_dense_1_nn_36_bias_read_readvariableop3
/savev2_dense_2_nn_36_kernel_read_readvariableop1
-savev2_dense_2_nn_36_bias_read_readvariableop3
/savev2_dense_3_nn_36_kernel_read_readvariableop1
-savev2_dense_3_nn_36_bias_read_readvariableop3
/savev2_dense_4_nn_36_kernel_read_readvariableop1
-savev2_dense_4_nn_36_bias_read_readvariableop3
/savev2_dense_5_nn_36_kernel_read_readvariableop1
-savev2_dense_5_nn_36_bias_read_readvariableop3
/savev2_dense_6_nn_36_kernel_read_readvariableop1
-savev2_dense_6_nn_36_bias_read_readvariableop3
/savev2_dense_7_nn_36_kernel_read_readvariableop1
-savev2_dense_7_nn_36_bias_read_readvariableop:
6savev2_dense_out_nl_8_nn_36_kernel_read_readvariableop8
4savev2_dense_out_nl_8_nn_36_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop:
6savev2_adam_dense_0_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_0_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_1_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_1_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_2_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_2_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_3_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_3_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_4_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_4_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_5_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_5_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_6_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_6_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_7_nn_36_kernel_m_read_readvariableop8
4savev2_adam_dense_7_nn_36_bias_m_read_readvariableopA
=savev2_adam_dense_out_nl_8_nn_36_kernel_m_read_readvariableop?
;savev2_adam_dense_out_nl_8_nn_36_bias_m_read_readvariableop:
6savev2_adam_dense_0_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_0_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_1_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_1_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_2_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_2_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_3_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_3_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_4_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_4_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_5_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_5_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_6_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_6_nn_36_bias_v_read_readvariableop:
6savev2_adam_dense_7_nn_36_kernel_v_read_readvariableop8
4savev2_adam_dense_7_nn_36_bias_v_read_readvariableopA
=savev2_adam_dense_out_nl_8_nn_36_kernel_v_read_readvariableop?
;savev2_adam_dense_out_nl_8_nn_36_bias_v_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�$
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�#
value�#B�#AB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop/savev2_dense_0_nn_36_kernel_read_readvariableop-savev2_dense_0_nn_36_bias_read_readvariableop/savev2_dense_1_nn_36_kernel_read_readvariableop-savev2_dense_1_nn_36_bias_read_readvariableop/savev2_dense_2_nn_36_kernel_read_readvariableop-savev2_dense_2_nn_36_bias_read_readvariableop/savev2_dense_3_nn_36_kernel_read_readvariableop-savev2_dense_3_nn_36_bias_read_readvariableop/savev2_dense_4_nn_36_kernel_read_readvariableop-savev2_dense_4_nn_36_bias_read_readvariableop/savev2_dense_5_nn_36_kernel_read_readvariableop-savev2_dense_5_nn_36_bias_read_readvariableop/savev2_dense_6_nn_36_kernel_read_readvariableop-savev2_dense_6_nn_36_bias_read_readvariableop/savev2_dense_7_nn_36_kernel_read_readvariableop-savev2_dense_7_nn_36_bias_read_readvariableop6savev2_dense_out_nl_8_nn_36_kernel_read_readvariableop4savev2_dense_out_nl_8_nn_36_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop6savev2_adam_dense_0_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_0_nn_36_bias_m_read_readvariableop6savev2_adam_dense_1_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_1_nn_36_bias_m_read_readvariableop6savev2_adam_dense_2_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_2_nn_36_bias_m_read_readvariableop6savev2_adam_dense_3_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_3_nn_36_bias_m_read_readvariableop6savev2_adam_dense_4_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_4_nn_36_bias_m_read_readvariableop6savev2_adam_dense_5_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_5_nn_36_bias_m_read_readvariableop6savev2_adam_dense_6_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_6_nn_36_bias_m_read_readvariableop6savev2_adam_dense_7_nn_36_kernel_m_read_readvariableop4savev2_adam_dense_7_nn_36_bias_m_read_readvariableop=savev2_adam_dense_out_nl_8_nn_36_kernel_m_read_readvariableop;savev2_adam_dense_out_nl_8_nn_36_bias_m_read_readvariableop6savev2_adam_dense_0_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_0_nn_36_bias_v_read_readvariableop6savev2_adam_dense_1_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_1_nn_36_bias_v_read_readvariableop6savev2_adam_dense_2_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_2_nn_36_bias_v_read_readvariableop6savev2_adam_dense_3_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_3_nn_36_bias_v_read_readvariableop6savev2_adam_dense_4_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_4_nn_36_bias_v_read_readvariableop6savev2_adam_dense_5_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_5_nn_36_bias_v_read_readvariableop6savev2_adam_dense_6_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_6_nn_36_bias_v_read_readvariableop6savev2_adam_dense_7_nn_36_kernel_v_read_readvariableop4savev2_adam_dense_7_nn_36_bias_v_read_readvariableop=savev2_adam_dense_out_nl_8_nn_36_kernel_v_read_readvariableop;savev2_adam_dense_out_nl_8_nn_36_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A		2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: ::: :$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$:: : : : : : : :$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$::$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$$:$:$:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :$ 

_output_shapes

:$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 	

_output_shapes
:$:$
 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$$: 

_output_shapes
:$:$ 

_output_shapes

:$: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:$: 

_output_shapes
:$:$ 

_output_shapes

:$$:  

_output_shapes
:$:$! 

_output_shapes

:$$: "

_output_shapes
:$:$# 

_output_shapes

:$$: $

_output_shapes
:$:$% 

_output_shapes

:$$: &

_output_shapes
:$:$' 

_output_shapes

:$$: (

_output_shapes
:$:$) 

_output_shapes

:$$: *

_output_shapes
:$:$+ 

_output_shapes

:$$: ,

_output_shapes
:$:$- 

_output_shapes

:$: .

_output_shapes
::$/ 

_output_shapes

:$: 0

_output_shapes
:$:$1 

_output_shapes

:$$: 2

_output_shapes
:$:$3 

_output_shapes

:$$: 4

_output_shapes
:$:$5 

_output_shapes

:$$: 6

_output_shapes
:$:$7 

_output_shapes

:$$: 8

_output_shapes
:$:$9 

_output_shapes

:$$: :

_output_shapes
:$:$; 

_output_shapes

:$$: <

_output_shapes
:$:$= 

_output_shapes

:$$: >

_output_shapes
:$:$? 

_output_shapes

:$: @

_output_shapes
::A

_output_shapes
: 
�	
�
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_2192906

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_1_nn_36_layer_call_fn_2193770

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_21928282
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_2193023

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_2192789

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_2193848

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_2193877

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_2193419
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_21933762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
,__inference_sequential_layer_call_fn_2193722

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_21933762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_2192966

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_2193775

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_7_nn_36_layer_call_fn_2193944

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_21930622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_2_nn_36_layer_call_fn_2193799

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_21928672
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_2193677

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_21932612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_2_nn_36_layer_call_fn_2193809

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_21928882
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_2193920

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�r
�
G__inference_sequential_layer_call_and_return_conditional_losses_2193553

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource0
,dense_0_nn_36_matmul_readvariableop_resource1
-dense_0_nn_36_biasadd_readvariableop_resource0
,dense_1_nn_36_matmul_readvariableop_resource1
-dense_1_nn_36_biasadd_readvariableop_resource0
,dense_2_nn_36_matmul_readvariableop_resource1
-dense_2_nn_36_biasadd_readvariableop_resource0
,dense_3_nn_36_matmul_readvariableop_resource1
-dense_3_nn_36_biasadd_readvariableop_resource0
,dense_4_nn_36_matmul_readvariableop_resource1
-dense_4_nn_36_biasadd_readvariableop_resource0
,dense_5_nn_36_matmul_readvariableop_resource1
-dense_5_nn_36_biasadd_readvariableop_resource0
,dense_6_nn_36_matmul_readvariableop_resource1
-dense_6_nn_36_biasadd_readvariableop_resource0
,dense_7_nn_36_matmul_readvariableop_resource1
-dense_7_nn_36_biasadd_readvariableop_resource7
3dense_out_nl_8_nn_36_matmul_readvariableop_resource8
4dense_out_nl_8_nn_36_biasadd_readvariableop_resource
identity��$dense_0_nn_36/BiasAdd/ReadVariableOp�#dense_0_nn_36/MatMul/ReadVariableOp�$dense_1_nn_36/BiasAdd/ReadVariableOp�#dense_1_nn_36/MatMul/ReadVariableOp�$dense_2_nn_36/BiasAdd/ReadVariableOp�#dense_2_nn_36/MatMul/ReadVariableOp�$dense_3_nn_36/BiasAdd/ReadVariableOp�#dense_3_nn_36/MatMul/ReadVariableOp�$dense_4_nn_36/BiasAdd/ReadVariableOp�#dense_4_nn_36/MatMul/ReadVariableOp�$dense_5_nn_36/BiasAdd/ReadVariableOp�#dense_5_nn_36/MatMul/ReadVariableOp�$dense_6_nn_36/BiasAdd/ReadVariableOp�#dense_6_nn_36/MatMul/ReadVariableOp�$dense_7_nn_36/BiasAdd/ReadVariableOp�#dense_7_nn_36/MatMul/ReadVariableOp�+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�*dense_out_nl_8_nn_36/MatMul/ReadVariableOp�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
#dense_0_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_0_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02%
#dense_0_nn_36/MatMul/ReadVariableOp�
dense_0_nn_36/MatMulMatMulnormalization/truediv:z:0+dense_0_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_0_nn_36/MatMul�
$dense_0_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_0_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_0_nn_36/BiasAdd/ReadVariableOp�
dense_0_nn_36/BiasAddBiasAdddense_0_nn_36/MatMul:product:0,dense_0_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_0_nn_36/BiasAdd�
leaky_re_lu_0_nn_36/LeakyRelu	LeakyReludense_0_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_0_nn_36/LeakyRelu�
#dense_1_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_1_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_1_nn_36/MatMul/ReadVariableOp�
dense_1_nn_36/MatMulMatMul+leaky_re_lu_0_nn_36/LeakyRelu:activations:0+dense_1_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1_nn_36/MatMul�
$dense_1_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_1_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_1_nn_36/BiasAdd/ReadVariableOp�
dense_1_nn_36/BiasAddBiasAdddense_1_nn_36/MatMul:product:0,dense_1_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1_nn_36/BiasAdd�
leaky_re_lu_1_nn_36/LeakyRelu	LeakyReludense_1_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_1_nn_36/LeakyRelu�
#dense_2_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_2_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_2_nn_36/MatMul/ReadVariableOp�
dense_2_nn_36/MatMulMatMul+leaky_re_lu_1_nn_36/LeakyRelu:activations:0+dense_2_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_2_nn_36/MatMul�
$dense_2_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_2_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_2_nn_36/BiasAdd/ReadVariableOp�
dense_2_nn_36/BiasAddBiasAdddense_2_nn_36/MatMul:product:0,dense_2_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_2_nn_36/BiasAdd�
leaky_re_lu_2_nn_36/LeakyRelu	LeakyReludense_2_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_2_nn_36/LeakyRelu�
#dense_3_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_3_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_3_nn_36/MatMul/ReadVariableOp�
dense_3_nn_36/MatMulMatMul+leaky_re_lu_2_nn_36/LeakyRelu:activations:0+dense_3_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_3_nn_36/MatMul�
$dense_3_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_3_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_3_nn_36/BiasAdd/ReadVariableOp�
dense_3_nn_36/BiasAddBiasAdddense_3_nn_36/MatMul:product:0,dense_3_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_3_nn_36/BiasAdd�
leaky_re_lu_3_nn_36/LeakyRelu	LeakyReludense_3_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_3_nn_36/LeakyRelu�
#dense_4_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_4_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_4_nn_36/MatMul/ReadVariableOp�
dense_4_nn_36/MatMulMatMul+leaky_re_lu_3_nn_36/LeakyRelu:activations:0+dense_4_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_4_nn_36/MatMul�
$dense_4_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_4_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_4_nn_36/BiasAdd/ReadVariableOp�
dense_4_nn_36/BiasAddBiasAdddense_4_nn_36/MatMul:product:0,dense_4_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_4_nn_36/BiasAdd�
leaky_re_lu_4_nn_36/LeakyRelu	LeakyReludense_4_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_4_nn_36/LeakyRelu�
#dense_5_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_5_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_5_nn_36/MatMul/ReadVariableOp�
dense_5_nn_36/MatMulMatMul+leaky_re_lu_4_nn_36/LeakyRelu:activations:0+dense_5_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_5_nn_36/MatMul�
$dense_5_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_5_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_5_nn_36/BiasAdd/ReadVariableOp�
dense_5_nn_36/BiasAddBiasAdddense_5_nn_36/MatMul:product:0,dense_5_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_5_nn_36/BiasAdd�
leaky_re_lu_5_nn_36/LeakyRelu	LeakyReludense_5_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_5_nn_36/LeakyRelu�
#dense_6_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_6_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_6_nn_36/MatMul/ReadVariableOp�
dense_6_nn_36/MatMulMatMul+leaky_re_lu_5_nn_36/LeakyRelu:activations:0+dense_6_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_6_nn_36/MatMul�
$dense_6_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_6_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_6_nn_36/BiasAdd/ReadVariableOp�
dense_6_nn_36/BiasAddBiasAdddense_6_nn_36/MatMul:product:0,dense_6_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_6_nn_36/BiasAdd�
leaky_re_lu_6_nn_36/LeakyRelu	LeakyReludense_6_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_6_nn_36/LeakyRelu�
#dense_7_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_7_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_7_nn_36/MatMul/ReadVariableOp�
dense_7_nn_36/MatMulMatMul+leaky_re_lu_6_nn_36/LeakyRelu:activations:0+dense_7_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_7_nn_36/MatMul�
$dense_7_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_7_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_7_nn_36/BiasAdd/ReadVariableOp�
dense_7_nn_36/BiasAddBiasAdddense_7_nn_36/MatMul:product:0,dense_7_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_7_nn_36/BiasAdd�
leaky_re_lu_7_nn_36/LeakyRelu	LeakyReludense_7_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_7_nn_36/LeakyRelu�
*dense_out_nl_8_nn_36/MatMul/ReadVariableOpReadVariableOp3dense_out_nl_8_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02,
*dense_out_nl_8_nn_36/MatMul/ReadVariableOp�
dense_out_nl_8_nn_36/MatMulMatMul+leaky_re_lu_7_nn_36/LeakyRelu:activations:02dense_out_nl_8_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_out_nl_8_nn_36/MatMul�
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOpReadVariableOp4dense_out_nl_8_nn_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�
dense_out_nl_8_nn_36/BiasAddBiasAdd%dense_out_nl_8_nn_36/MatMul:product:03dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_out_nl_8_nn_36/BiasAdd�
IdentityIdentity%dense_out_nl_8_nn_36/BiasAdd:output:0%^dense_0_nn_36/BiasAdd/ReadVariableOp$^dense_0_nn_36/MatMul/ReadVariableOp%^dense_1_nn_36/BiasAdd/ReadVariableOp$^dense_1_nn_36/MatMul/ReadVariableOp%^dense_2_nn_36/BiasAdd/ReadVariableOp$^dense_2_nn_36/MatMul/ReadVariableOp%^dense_3_nn_36/BiasAdd/ReadVariableOp$^dense_3_nn_36/MatMul/ReadVariableOp%^dense_4_nn_36/BiasAdd/ReadVariableOp$^dense_4_nn_36/MatMul/ReadVariableOp%^dense_5_nn_36/BiasAdd/ReadVariableOp$^dense_5_nn_36/MatMul/ReadVariableOp%^dense_6_nn_36/BiasAdd/ReadVariableOp$^dense_6_nn_36/MatMul/ReadVariableOp%^dense_7_nn_36/BiasAdd/ReadVariableOp$^dense_7_nn_36/MatMul/ReadVariableOp,^dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp+^dense_out_nl_8_nn_36/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2L
$dense_0_nn_36/BiasAdd/ReadVariableOp$dense_0_nn_36/BiasAdd/ReadVariableOp2J
#dense_0_nn_36/MatMul/ReadVariableOp#dense_0_nn_36/MatMul/ReadVariableOp2L
$dense_1_nn_36/BiasAdd/ReadVariableOp$dense_1_nn_36/BiasAdd/ReadVariableOp2J
#dense_1_nn_36/MatMul/ReadVariableOp#dense_1_nn_36/MatMul/ReadVariableOp2L
$dense_2_nn_36/BiasAdd/ReadVariableOp$dense_2_nn_36/BiasAdd/ReadVariableOp2J
#dense_2_nn_36/MatMul/ReadVariableOp#dense_2_nn_36/MatMul/ReadVariableOp2L
$dense_3_nn_36/BiasAdd/ReadVariableOp$dense_3_nn_36/BiasAdd/ReadVariableOp2J
#dense_3_nn_36/MatMul/ReadVariableOp#dense_3_nn_36/MatMul/ReadVariableOp2L
$dense_4_nn_36/BiasAdd/ReadVariableOp$dense_4_nn_36/BiasAdd/ReadVariableOp2J
#dense_4_nn_36/MatMul/ReadVariableOp#dense_4_nn_36/MatMul/ReadVariableOp2L
$dense_5_nn_36/BiasAdd/ReadVariableOp$dense_5_nn_36/BiasAdd/ReadVariableOp2J
#dense_5_nn_36/MatMul/ReadVariableOp#dense_5_nn_36/MatMul/ReadVariableOp2L
$dense_6_nn_36/BiasAdd/ReadVariableOp$dense_6_nn_36/BiasAdd/ReadVariableOp2J
#dense_6_nn_36/MatMul/ReadVariableOp#dense_6_nn_36/MatMul/ReadVariableOp2L
$dense_7_nn_36/BiasAdd/ReadVariableOp$dense_7_nn_36/BiasAdd/ReadVariableOp2J
#dense_7_nn_36/MatMul/ReadVariableOp#dense_7_nn_36/MatMul/ReadVariableOp2Z
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp2X
*dense_out_nl_8_nn_36/MatMul/ReadVariableOp*dense_out_nl_8_nn_36/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_7_nn_36_layer_call_fn_2193954

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_21930832
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_2193062

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_2192810

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_5_nn_36_layer_call_fn_2193896

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_21930052
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_6_nn_36_layer_call_fn_2193925

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_21930442
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_3_nn_36_layer_call_fn_2193828

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_21929062
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_2192984

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�a
�	
G__inference_sequential_layer_call_and_return_conditional_losses_2193118
normalization_input1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_0_nn_36_2192800
dense_0_nn_36_2192802
dense_1_nn_36_2192839
dense_1_nn_36_2192841
dense_2_nn_36_2192878
dense_2_nn_36_2192880
dense_3_nn_36_2192917
dense_3_nn_36_2192919
dense_4_nn_36_2192956
dense_4_nn_36_2192958
dense_5_nn_36_2192995
dense_5_nn_36_2192997
dense_6_nn_36_2193034
dense_6_nn_36_2193036
dense_7_nn_36_2193073
dense_7_nn_36_2193075 
dense_out_nl_8_nn_36_2193112 
dense_out_nl_8_nn_36_2193114
identity��%dense_0_nn_36/StatefulPartitionedCall�%dense_1_nn_36/StatefulPartitionedCall�%dense_2_nn_36/StatefulPartitionedCall�%dense_3_nn_36/StatefulPartitionedCall�%dense_4_nn_36/StatefulPartitionedCall�%dense_5_nn_36/StatefulPartitionedCall�%dense_6_nn_36/StatefulPartitionedCall�%dense_7_nn_36/StatefulPartitionedCall�,dense_out_nl_8_nn_36/StatefulPartitionedCall�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubnormalization_inputnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
%dense_0_nn_36/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_0_nn_36_2192800dense_0_nn_36_2192802*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_21927892'
%dense_0_nn_36/StatefulPartitionedCall�
#leaky_re_lu_0_nn_36/PartitionedCallPartitionedCall.dense_0_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_21928102%
#leaky_re_lu_0_nn_36/PartitionedCall�
%dense_1_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_0_nn_36/PartitionedCall:output:0dense_1_nn_36_2192839dense_1_nn_36_2192841*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_21928282'
%dense_1_nn_36/StatefulPartitionedCall�
#leaky_re_lu_1_nn_36/PartitionedCallPartitionedCall.dense_1_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_21928492%
#leaky_re_lu_1_nn_36/PartitionedCall�
%dense_2_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_1_nn_36/PartitionedCall:output:0dense_2_nn_36_2192878dense_2_nn_36_2192880*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_21928672'
%dense_2_nn_36/StatefulPartitionedCall�
#leaky_re_lu_2_nn_36/PartitionedCallPartitionedCall.dense_2_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_21928882%
#leaky_re_lu_2_nn_36/PartitionedCall�
%dense_3_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_2_nn_36/PartitionedCall:output:0dense_3_nn_36_2192917dense_3_nn_36_2192919*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_21929062'
%dense_3_nn_36/StatefulPartitionedCall�
#leaky_re_lu_3_nn_36/PartitionedCallPartitionedCall.dense_3_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_21929272%
#leaky_re_lu_3_nn_36/PartitionedCall�
%dense_4_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_3_nn_36/PartitionedCall:output:0dense_4_nn_36_2192956dense_4_nn_36_2192958*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_21929452'
%dense_4_nn_36/StatefulPartitionedCall�
#leaky_re_lu_4_nn_36/PartitionedCallPartitionedCall.dense_4_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_21929662%
#leaky_re_lu_4_nn_36/PartitionedCall�
%dense_5_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_4_nn_36/PartitionedCall:output:0dense_5_nn_36_2192995dense_5_nn_36_2192997*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_21929842'
%dense_5_nn_36/StatefulPartitionedCall�
#leaky_re_lu_5_nn_36/PartitionedCallPartitionedCall.dense_5_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_21930052%
#leaky_re_lu_5_nn_36/PartitionedCall�
%dense_6_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_5_nn_36/PartitionedCall:output:0dense_6_nn_36_2193034dense_6_nn_36_2193036*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_21930232'
%dense_6_nn_36/StatefulPartitionedCall�
#leaky_re_lu_6_nn_36/PartitionedCallPartitionedCall.dense_6_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_21930442%
#leaky_re_lu_6_nn_36/PartitionedCall�
%dense_7_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_6_nn_36/PartitionedCall:output:0dense_7_nn_36_2193073dense_7_nn_36_2193075*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_21930622'
%dense_7_nn_36/StatefulPartitionedCall�
#leaky_re_lu_7_nn_36/PartitionedCallPartitionedCall.dense_7_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_21930832%
#leaky_re_lu_7_nn_36/PartitionedCall�
,dense_out_nl_8_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_7_nn_36/PartitionedCall:output:0dense_out_nl_8_nn_36_2193112dense_out_nl_8_nn_36_2193114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_21931012.
,dense_out_nl_8_nn_36/StatefulPartitionedCall�
IdentityIdentity5dense_out_nl_8_nn_36/StatefulPartitionedCall:output:0&^dense_0_nn_36/StatefulPartitionedCall&^dense_1_nn_36/StatefulPartitionedCall&^dense_2_nn_36/StatefulPartitionedCall&^dense_3_nn_36/StatefulPartitionedCall&^dense_4_nn_36/StatefulPartitionedCall&^dense_5_nn_36/StatefulPartitionedCall&^dense_6_nn_36/StatefulPartitionedCall&^dense_7_nn_36/StatefulPartitionedCall-^dense_out_nl_8_nn_36/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2N
%dense_0_nn_36/StatefulPartitionedCall%dense_0_nn_36/StatefulPartitionedCall2N
%dense_1_nn_36/StatefulPartitionedCall%dense_1_nn_36/StatefulPartitionedCall2N
%dense_2_nn_36/StatefulPartitionedCall%dense_2_nn_36/StatefulPartitionedCall2N
%dense_3_nn_36/StatefulPartitionedCall%dense_3_nn_36/StatefulPartitionedCall2N
%dense_4_nn_36/StatefulPartitionedCall%dense_4_nn_36/StatefulPartitionedCall2N
%dense_5_nn_36/StatefulPartitionedCall%dense_5_nn_36/StatefulPartitionedCall2N
%dense_6_nn_36/StatefulPartitionedCall%dense_6_nn_36/StatefulPartitionedCall2N
%dense_7_nn_36/StatefulPartitionedCall%dense_7_nn_36/StatefulPartitionedCall2\
,dense_out_nl_8_nn_36/StatefulPartitionedCall,dense_out_nl_8_nn_36/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�r
�
G__inference_sequential_layer_call_and_return_conditional_losses_2193632

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource0
,dense_0_nn_36_matmul_readvariableop_resource1
-dense_0_nn_36_biasadd_readvariableop_resource0
,dense_1_nn_36_matmul_readvariableop_resource1
-dense_1_nn_36_biasadd_readvariableop_resource0
,dense_2_nn_36_matmul_readvariableop_resource1
-dense_2_nn_36_biasadd_readvariableop_resource0
,dense_3_nn_36_matmul_readvariableop_resource1
-dense_3_nn_36_biasadd_readvariableop_resource0
,dense_4_nn_36_matmul_readvariableop_resource1
-dense_4_nn_36_biasadd_readvariableop_resource0
,dense_5_nn_36_matmul_readvariableop_resource1
-dense_5_nn_36_biasadd_readvariableop_resource0
,dense_6_nn_36_matmul_readvariableop_resource1
-dense_6_nn_36_biasadd_readvariableop_resource0
,dense_7_nn_36_matmul_readvariableop_resource1
-dense_7_nn_36_biasadd_readvariableop_resource7
3dense_out_nl_8_nn_36_matmul_readvariableop_resource8
4dense_out_nl_8_nn_36_biasadd_readvariableop_resource
identity��$dense_0_nn_36/BiasAdd/ReadVariableOp�#dense_0_nn_36/MatMul/ReadVariableOp�$dense_1_nn_36/BiasAdd/ReadVariableOp�#dense_1_nn_36/MatMul/ReadVariableOp�$dense_2_nn_36/BiasAdd/ReadVariableOp�#dense_2_nn_36/MatMul/ReadVariableOp�$dense_3_nn_36/BiasAdd/ReadVariableOp�#dense_3_nn_36/MatMul/ReadVariableOp�$dense_4_nn_36/BiasAdd/ReadVariableOp�#dense_4_nn_36/MatMul/ReadVariableOp�$dense_5_nn_36/BiasAdd/ReadVariableOp�#dense_5_nn_36/MatMul/ReadVariableOp�$dense_6_nn_36/BiasAdd/ReadVariableOp�#dense_6_nn_36/MatMul/ReadVariableOp�$dense_7_nn_36/BiasAdd/ReadVariableOp�#dense_7_nn_36/MatMul/ReadVariableOp�+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�*dense_out_nl_8_nn_36/MatMul/ReadVariableOp�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
#dense_0_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_0_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02%
#dense_0_nn_36/MatMul/ReadVariableOp�
dense_0_nn_36/MatMulMatMulnormalization/truediv:z:0+dense_0_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_0_nn_36/MatMul�
$dense_0_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_0_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_0_nn_36/BiasAdd/ReadVariableOp�
dense_0_nn_36/BiasAddBiasAdddense_0_nn_36/MatMul:product:0,dense_0_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_0_nn_36/BiasAdd�
leaky_re_lu_0_nn_36/LeakyRelu	LeakyReludense_0_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_0_nn_36/LeakyRelu�
#dense_1_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_1_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_1_nn_36/MatMul/ReadVariableOp�
dense_1_nn_36/MatMulMatMul+leaky_re_lu_0_nn_36/LeakyRelu:activations:0+dense_1_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1_nn_36/MatMul�
$dense_1_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_1_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_1_nn_36/BiasAdd/ReadVariableOp�
dense_1_nn_36/BiasAddBiasAdddense_1_nn_36/MatMul:product:0,dense_1_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_1_nn_36/BiasAdd�
leaky_re_lu_1_nn_36/LeakyRelu	LeakyReludense_1_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_1_nn_36/LeakyRelu�
#dense_2_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_2_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_2_nn_36/MatMul/ReadVariableOp�
dense_2_nn_36/MatMulMatMul+leaky_re_lu_1_nn_36/LeakyRelu:activations:0+dense_2_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_2_nn_36/MatMul�
$dense_2_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_2_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_2_nn_36/BiasAdd/ReadVariableOp�
dense_2_nn_36/BiasAddBiasAdddense_2_nn_36/MatMul:product:0,dense_2_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_2_nn_36/BiasAdd�
leaky_re_lu_2_nn_36/LeakyRelu	LeakyReludense_2_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_2_nn_36/LeakyRelu�
#dense_3_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_3_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_3_nn_36/MatMul/ReadVariableOp�
dense_3_nn_36/MatMulMatMul+leaky_re_lu_2_nn_36/LeakyRelu:activations:0+dense_3_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_3_nn_36/MatMul�
$dense_3_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_3_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_3_nn_36/BiasAdd/ReadVariableOp�
dense_3_nn_36/BiasAddBiasAdddense_3_nn_36/MatMul:product:0,dense_3_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_3_nn_36/BiasAdd�
leaky_re_lu_3_nn_36/LeakyRelu	LeakyReludense_3_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_3_nn_36/LeakyRelu�
#dense_4_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_4_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_4_nn_36/MatMul/ReadVariableOp�
dense_4_nn_36/MatMulMatMul+leaky_re_lu_3_nn_36/LeakyRelu:activations:0+dense_4_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_4_nn_36/MatMul�
$dense_4_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_4_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_4_nn_36/BiasAdd/ReadVariableOp�
dense_4_nn_36/BiasAddBiasAdddense_4_nn_36/MatMul:product:0,dense_4_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_4_nn_36/BiasAdd�
leaky_re_lu_4_nn_36/LeakyRelu	LeakyReludense_4_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_4_nn_36/LeakyRelu�
#dense_5_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_5_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_5_nn_36/MatMul/ReadVariableOp�
dense_5_nn_36/MatMulMatMul+leaky_re_lu_4_nn_36/LeakyRelu:activations:0+dense_5_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_5_nn_36/MatMul�
$dense_5_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_5_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_5_nn_36/BiasAdd/ReadVariableOp�
dense_5_nn_36/BiasAddBiasAdddense_5_nn_36/MatMul:product:0,dense_5_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_5_nn_36/BiasAdd�
leaky_re_lu_5_nn_36/LeakyRelu	LeakyReludense_5_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_5_nn_36/LeakyRelu�
#dense_6_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_6_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_6_nn_36/MatMul/ReadVariableOp�
dense_6_nn_36/MatMulMatMul+leaky_re_lu_5_nn_36/LeakyRelu:activations:0+dense_6_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_6_nn_36/MatMul�
$dense_6_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_6_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_6_nn_36/BiasAdd/ReadVariableOp�
dense_6_nn_36/BiasAddBiasAdddense_6_nn_36/MatMul:product:0,dense_6_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_6_nn_36/BiasAdd�
leaky_re_lu_6_nn_36/LeakyRelu	LeakyReludense_6_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_6_nn_36/LeakyRelu�
#dense_7_nn_36/MatMul/ReadVariableOpReadVariableOp,dense_7_nn_36_matmul_readvariableop_resource*
_output_shapes

:$$*
dtype02%
#dense_7_nn_36/MatMul/ReadVariableOp�
dense_7_nn_36/MatMulMatMul+leaky_re_lu_6_nn_36/LeakyRelu:activations:0+dense_7_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_7_nn_36/MatMul�
$dense_7_nn_36/BiasAdd/ReadVariableOpReadVariableOp-dense_7_nn_36_biasadd_readvariableop_resource*
_output_shapes
:$*
dtype02&
$dense_7_nn_36/BiasAdd/ReadVariableOp�
dense_7_nn_36/BiasAddBiasAdddense_7_nn_36/MatMul:product:0,dense_7_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
dense_7_nn_36/BiasAdd�
leaky_re_lu_7_nn_36/LeakyRelu	LeakyReludense_7_nn_36/BiasAdd:output:0*'
_output_shapes
:���������$*
alpha%���=2
leaky_re_lu_7_nn_36/LeakyRelu�
*dense_out_nl_8_nn_36/MatMul/ReadVariableOpReadVariableOp3dense_out_nl_8_nn_36_matmul_readvariableop_resource*
_output_shapes

:$*
dtype02,
*dense_out_nl_8_nn_36/MatMul/ReadVariableOp�
dense_out_nl_8_nn_36/MatMulMatMul+leaky_re_lu_7_nn_36/LeakyRelu:activations:02dense_out_nl_8_nn_36/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_out_nl_8_nn_36/MatMul�
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOpReadVariableOp4dense_out_nl_8_nn_36_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp�
dense_out_nl_8_nn_36/BiasAddBiasAdd%dense_out_nl_8_nn_36/MatMul:product:03dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
dense_out_nl_8_nn_36/BiasAdd�
IdentityIdentity%dense_out_nl_8_nn_36/BiasAdd:output:0%^dense_0_nn_36/BiasAdd/ReadVariableOp$^dense_0_nn_36/MatMul/ReadVariableOp%^dense_1_nn_36/BiasAdd/ReadVariableOp$^dense_1_nn_36/MatMul/ReadVariableOp%^dense_2_nn_36/BiasAdd/ReadVariableOp$^dense_2_nn_36/MatMul/ReadVariableOp%^dense_3_nn_36/BiasAdd/ReadVariableOp$^dense_3_nn_36/MatMul/ReadVariableOp%^dense_4_nn_36/BiasAdd/ReadVariableOp$^dense_4_nn_36/MatMul/ReadVariableOp%^dense_5_nn_36/BiasAdd/ReadVariableOp$^dense_5_nn_36/MatMul/ReadVariableOp%^dense_6_nn_36/BiasAdd/ReadVariableOp$^dense_6_nn_36/MatMul/ReadVariableOp%^dense_7_nn_36/BiasAdd/ReadVariableOp$^dense_7_nn_36/MatMul/ReadVariableOp,^dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp+^dense_out_nl_8_nn_36/MatMul/ReadVariableOp%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2L
$dense_0_nn_36/BiasAdd/ReadVariableOp$dense_0_nn_36/BiasAdd/ReadVariableOp2J
#dense_0_nn_36/MatMul/ReadVariableOp#dense_0_nn_36/MatMul/ReadVariableOp2L
$dense_1_nn_36/BiasAdd/ReadVariableOp$dense_1_nn_36/BiasAdd/ReadVariableOp2J
#dense_1_nn_36/MatMul/ReadVariableOp#dense_1_nn_36/MatMul/ReadVariableOp2L
$dense_2_nn_36/BiasAdd/ReadVariableOp$dense_2_nn_36/BiasAdd/ReadVariableOp2J
#dense_2_nn_36/MatMul/ReadVariableOp#dense_2_nn_36/MatMul/ReadVariableOp2L
$dense_3_nn_36/BiasAdd/ReadVariableOp$dense_3_nn_36/BiasAdd/ReadVariableOp2J
#dense_3_nn_36/MatMul/ReadVariableOp#dense_3_nn_36/MatMul/ReadVariableOp2L
$dense_4_nn_36/BiasAdd/ReadVariableOp$dense_4_nn_36/BiasAdd/ReadVariableOp2J
#dense_4_nn_36/MatMul/ReadVariableOp#dense_4_nn_36/MatMul/ReadVariableOp2L
$dense_5_nn_36/BiasAdd/ReadVariableOp$dense_5_nn_36/BiasAdd/ReadVariableOp2J
#dense_5_nn_36/MatMul/ReadVariableOp#dense_5_nn_36/MatMul/ReadVariableOp2L
$dense_6_nn_36/BiasAdd/ReadVariableOp$dense_6_nn_36/BiasAdd/ReadVariableOp2J
#dense_6_nn_36/MatMul/ReadVariableOp#dense_6_nn_36/MatMul/ReadVariableOp2L
$dense_7_nn_36/BiasAdd/ReadVariableOp$dense_7_nn_36/BiasAdd/ReadVariableOp2J
#dense_7_nn_36/MatMul/ReadVariableOp#dense_7_nn_36/MatMul/ReadVariableOp2Z
+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp+dense_out_nl_8_nn_36/BiasAdd/ReadVariableOp2X
*dense_out_nl_8_nn_36/MatMul/ReadVariableOp*dense_out_nl_8_nn_36/MatMul/ReadVariableOp2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_2193083

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_5_nn_36_layer_call_fn_2193886

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_21929842
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_2193101

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
,__inference_sequential_layer_call_fn_2193304
normalization_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallnormalization_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*6
_read_only_resource_inputs
	
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *P
fKRI
G__inference_sequential_layer_call_and_return_conditional_losses_21932612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�
�
/__inference_dense_6_nn_36_layer_call_fn_2193915

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_21930232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_2193833

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_2192828

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
�
/__inference_dense_0_nn_36_layer_call_fn_2193741

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_21927892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_2193819

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_3_nn_36_layer_call_fn_2193838

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_21929272
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_2193804

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_2192849

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_2193746

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
��
�#
#__inference__traced_restore_2194390
file_prefix
assignvariableop_mean
assignvariableop_1_variance
assignvariableop_2_count+
'assignvariableop_3_dense_0_nn_36_kernel)
%assignvariableop_4_dense_0_nn_36_bias+
'assignvariableop_5_dense_1_nn_36_kernel)
%assignvariableop_6_dense_1_nn_36_bias+
'assignvariableop_7_dense_2_nn_36_kernel)
%assignvariableop_8_dense_2_nn_36_bias+
'assignvariableop_9_dense_3_nn_36_kernel*
&assignvariableop_10_dense_3_nn_36_bias,
(assignvariableop_11_dense_4_nn_36_kernel*
&assignvariableop_12_dense_4_nn_36_bias,
(assignvariableop_13_dense_5_nn_36_kernel*
&assignvariableop_14_dense_5_nn_36_bias,
(assignvariableop_15_dense_6_nn_36_kernel*
&assignvariableop_16_dense_6_nn_36_bias,
(assignvariableop_17_dense_7_nn_36_kernel*
&assignvariableop_18_dense_7_nn_36_bias3
/assignvariableop_19_dense_out_nl_8_nn_36_kernel1
-assignvariableop_20_dense_out_nl_8_nn_36_bias!
assignvariableop_21_adam_iter#
assignvariableop_22_adam_beta_1#
assignvariableop_23_adam_beta_2"
assignvariableop_24_adam_decay*
&assignvariableop_25_adam_learning_rate
assignvariableop_26_total
assignvariableop_27_count_13
/assignvariableop_28_adam_dense_0_nn_36_kernel_m1
-assignvariableop_29_adam_dense_0_nn_36_bias_m3
/assignvariableop_30_adam_dense_1_nn_36_kernel_m1
-assignvariableop_31_adam_dense_1_nn_36_bias_m3
/assignvariableop_32_adam_dense_2_nn_36_kernel_m1
-assignvariableop_33_adam_dense_2_nn_36_bias_m3
/assignvariableop_34_adam_dense_3_nn_36_kernel_m1
-assignvariableop_35_adam_dense_3_nn_36_bias_m3
/assignvariableop_36_adam_dense_4_nn_36_kernel_m1
-assignvariableop_37_adam_dense_4_nn_36_bias_m3
/assignvariableop_38_adam_dense_5_nn_36_kernel_m1
-assignvariableop_39_adam_dense_5_nn_36_bias_m3
/assignvariableop_40_adam_dense_6_nn_36_kernel_m1
-assignvariableop_41_adam_dense_6_nn_36_bias_m3
/assignvariableop_42_adam_dense_7_nn_36_kernel_m1
-assignvariableop_43_adam_dense_7_nn_36_bias_m:
6assignvariableop_44_adam_dense_out_nl_8_nn_36_kernel_m8
4assignvariableop_45_adam_dense_out_nl_8_nn_36_bias_m3
/assignvariableop_46_adam_dense_0_nn_36_kernel_v1
-assignvariableop_47_adam_dense_0_nn_36_bias_v3
/assignvariableop_48_adam_dense_1_nn_36_kernel_v1
-assignvariableop_49_adam_dense_1_nn_36_bias_v3
/assignvariableop_50_adam_dense_2_nn_36_kernel_v1
-assignvariableop_51_adam_dense_2_nn_36_bias_v3
/assignvariableop_52_adam_dense_3_nn_36_kernel_v1
-assignvariableop_53_adam_dense_3_nn_36_bias_v3
/assignvariableop_54_adam_dense_4_nn_36_kernel_v1
-assignvariableop_55_adam_dense_4_nn_36_bias_v3
/assignvariableop_56_adam_dense_5_nn_36_kernel_v1
-assignvariableop_57_adam_dense_5_nn_36_bias_v3
/assignvariableop_58_adam_dense_6_nn_36_kernel_v1
-assignvariableop_59_adam_dense_6_nn_36_bias_v3
/assignvariableop_60_adam_dense_7_nn_36_kernel_v1
-assignvariableop_61_adam_dense_7_nn_36_bias_v:
6assignvariableop_62_adam_dense_out_nl_8_nn_36_kernel_v8
4assignvariableop_63_adam_dense_out_nl_8_nn_36_bias_v
identity_65��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_54�AssignVariableOp_55�AssignVariableOp_56�AssignVariableOp_57�AssignVariableOp_58�AssignVariableOp_59�AssignVariableOp_6�AssignVariableOp_60�AssignVariableOp_61�AssignVariableOp_62�AssignVariableOp_63�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�$
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�#
value�#B�#AB4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-5/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-7/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*�
value�B�AB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A		2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp'assignvariableop_3_dense_0_nn_36_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp%assignvariableop_4_dense_0_nn_36_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp'assignvariableop_5_dense_1_nn_36_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp%assignvariableop_6_dense_1_nn_36_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp'assignvariableop_7_dense_2_nn_36_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp%assignvariableop_8_dense_2_nn_36_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp'assignvariableop_9_dense_3_nn_36_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp&assignvariableop_10_dense_3_nn_36_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp(assignvariableop_11_dense_4_nn_36_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp&assignvariableop_12_dense_4_nn_36_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp(assignvariableop_13_dense_5_nn_36_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp&assignvariableop_14_dense_5_nn_36_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp(assignvariableop_15_dense_6_nn_36_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp&assignvariableop_16_dense_6_nn_36_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp(assignvariableop_17_dense_7_nn_36_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp&assignvariableop_18_dense_7_nn_36_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp/assignvariableop_19_dense_out_nl_8_nn_36_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp-assignvariableop_20_dense_out_nl_8_nn_36_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOpassignvariableop_21_adam_iterIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOpassignvariableop_22_adam_beta_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOpassignvariableop_23_adam_beta_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOpassignvariableop_24_adam_decayIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_learning_rateIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOpassignvariableop_26_totalIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOpassignvariableop_27_count_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp/assignvariableop_28_adam_dense_0_nn_36_kernel_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp-assignvariableop_29_adam_dense_0_nn_36_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp/assignvariableop_30_adam_dense_1_nn_36_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_dense_1_nn_36_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp/assignvariableop_32_adam_dense_2_nn_36_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_dense_2_nn_36_bias_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp/assignvariableop_34_adam_dense_3_nn_36_kernel_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_dense_3_nn_36_bias_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_adam_dense_4_nn_36_kernel_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp-assignvariableop_37_adam_dense_4_nn_36_bias_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp/assignvariableop_38_adam_dense_5_nn_36_kernel_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp-assignvariableop_39_adam_dense_5_nn_36_bias_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp/assignvariableop_40_adam_dense_6_nn_36_kernel_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp-assignvariableop_41_adam_dense_6_nn_36_bias_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp/assignvariableop_42_adam_dense_7_nn_36_kernel_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp-assignvariableop_43_adam_dense_7_nn_36_bias_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp6assignvariableop_44_adam_dense_out_nl_8_nn_36_kernel_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp4assignvariableop_45_adam_dense_out_nl_8_nn_36_bias_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp/assignvariableop_46_adam_dense_0_nn_36_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp-assignvariableop_47_adam_dense_0_nn_36_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp/assignvariableop_48_adam_dense_1_nn_36_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp-assignvariableop_49_adam_dense_1_nn_36_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp/assignvariableop_50_adam_dense_2_nn_36_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp-assignvariableop_51_adam_dense_2_nn_36_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp/assignvariableop_52_adam_dense_3_nn_36_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp-assignvariableop_53_adam_dense_3_nn_36_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54�
AssignVariableOp_54AssignVariableOp/assignvariableop_54_adam_dense_4_nn_36_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55�
AssignVariableOp_55AssignVariableOp-assignvariableop_55_adam_dense_4_nn_36_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56�
AssignVariableOp_56AssignVariableOp/assignvariableop_56_adam_dense_5_nn_36_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57�
AssignVariableOp_57AssignVariableOp-assignvariableop_57_adam_dense_5_nn_36_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58�
AssignVariableOp_58AssignVariableOp/assignvariableop_58_adam_dense_6_nn_36_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59�
AssignVariableOp_59AssignVariableOp-assignvariableop_59_adam_dense_6_nn_36_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60�
AssignVariableOp_60AssignVariableOp/assignvariableop_60_adam_dense_7_nn_36_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61�
AssignVariableOp_61AssignVariableOp-assignvariableop_61_adam_dense_7_nn_36_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62�
AssignVariableOp_62AssignVariableOp6assignvariableop_62_adam_dense_out_nl_8_nn_36_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63�
AssignVariableOp_63AssignVariableOp4assignvariableop_63_adam_dense_out_nl_8_nn_36_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64�
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�
l
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_2193005

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_2193732

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
6__inference_dense_out_nl_8_nn_36_layer_call_fn_2193973

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_21931012
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�a
�	
G__inference_sequential_layer_call_and_return_conditional_losses_2193188
normalization_input1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_0_nn_36_2193134
dense_0_nn_36_2193136
dense_1_nn_36_2193140
dense_1_nn_36_2193142
dense_2_nn_36_2193146
dense_2_nn_36_2193148
dense_3_nn_36_2193152
dense_3_nn_36_2193154
dense_4_nn_36_2193158
dense_4_nn_36_2193160
dense_5_nn_36_2193164
dense_5_nn_36_2193166
dense_6_nn_36_2193170
dense_6_nn_36_2193172
dense_7_nn_36_2193176
dense_7_nn_36_2193178 
dense_out_nl_8_nn_36_2193182 
dense_out_nl_8_nn_36_2193184
identity��%dense_0_nn_36/StatefulPartitionedCall�%dense_1_nn_36/StatefulPartitionedCall�%dense_2_nn_36/StatefulPartitionedCall�%dense_3_nn_36/StatefulPartitionedCall�%dense_4_nn_36/StatefulPartitionedCall�%dense_5_nn_36/StatefulPartitionedCall�%dense_6_nn_36/StatefulPartitionedCall�%dense_7_nn_36/StatefulPartitionedCall�,dense_out_nl_8_nn_36/StatefulPartitionedCall�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubnormalization_inputnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
%dense_0_nn_36/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_0_nn_36_2193134dense_0_nn_36_2193136*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_21927892'
%dense_0_nn_36/StatefulPartitionedCall�
#leaky_re_lu_0_nn_36/PartitionedCallPartitionedCall.dense_0_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_21928102%
#leaky_re_lu_0_nn_36/PartitionedCall�
%dense_1_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_0_nn_36/PartitionedCall:output:0dense_1_nn_36_2193140dense_1_nn_36_2193142*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_21928282'
%dense_1_nn_36/StatefulPartitionedCall�
#leaky_re_lu_1_nn_36/PartitionedCallPartitionedCall.dense_1_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_21928492%
#leaky_re_lu_1_nn_36/PartitionedCall�
%dense_2_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_1_nn_36/PartitionedCall:output:0dense_2_nn_36_2193146dense_2_nn_36_2193148*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_21928672'
%dense_2_nn_36/StatefulPartitionedCall�
#leaky_re_lu_2_nn_36/PartitionedCallPartitionedCall.dense_2_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_21928882%
#leaky_re_lu_2_nn_36/PartitionedCall�
%dense_3_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_2_nn_36/PartitionedCall:output:0dense_3_nn_36_2193152dense_3_nn_36_2193154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_21929062'
%dense_3_nn_36/StatefulPartitionedCall�
#leaky_re_lu_3_nn_36/PartitionedCallPartitionedCall.dense_3_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_21929272%
#leaky_re_lu_3_nn_36/PartitionedCall�
%dense_4_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_3_nn_36/PartitionedCall:output:0dense_4_nn_36_2193158dense_4_nn_36_2193160*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_21929452'
%dense_4_nn_36/StatefulPartitionedCall�
#leaky_re_lu_4_nn_36/PartitionedCallPartitionedCall.dense_4_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_21929662%
#leaky_re_lu_4_nn_36/PartitionedCall�
%dense_5_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_4_nn_36/PartitionedCall:output:0dense_5_nn_36_2193164dense_5_nn_36_2193166*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_21929842'
%dense_5_nn_36/StatefulPartitionedCall�
#leaky_re_lu_5_nn_36/PartitionedCallPartitionedCall.dense_5_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_21930052%
#leaky_re_lu_5_nn_36/PartitionedCall�
%dense_6_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_5_nn_36/PartitionedCall:output:0dense_6_nn_36_2193170dense_6_nn_36_2193172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_21930232'
%dense_6_nn_36/StatefulPartitionedCall�
#leaky_re_lu_6_nn_36/PartitionedCallPartitionedCall.dense_6_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_21930442%
#leaky_re_lu_6_nn_36/PartitionedCall�
%dense_7_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_6_nn_36/PartitionedCall:output:0dense_7_nn_36_2193176dense_7_nn_36_2193178*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_21930622'
%dense_7_nn_36/StatefulPartitionedCall�
#leaky_re_lu_7_nn_36/PartitionedCallPartitionedCall.dense_7_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_21930832%
#leaky_re_lu_7_nn_36/PartitionedCall�
,dense_out_nl_8_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_7_nn_36/PartitionedCall:output:0dense_out_nl_8_nn_36_2193182dense_out_nl_8_nn_36_2193184*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_21931012.
,dense_out_nl_8_nn_36/StatefulPartitionedCall�
IdentityIdentity5dense_out_nl_8_nn_36/StatefulPartitionedCall:output:0&^dense_0_nn_36/StatefulPartitionedCall&^dense_1_nn_36/StatefulPartitionedCall&^dense_2_nn_36/StatefulPartitionedCall&^dense_3_nn_36/StatefulPartitionedCall&^dense_4_nn_36/StatefulPartitionedCall&^dense_5_nn_36/StatefulPartitionedCall&^dense_6_nn_36/StatefulPartitionedCall&^dense_7_nn_36/StatefulPartitionedCall-^dense_out_nl_8_nn_36/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2N
%dense_0_nn_36/StatefulPartitionedCall%dense_0_nn_36/StatefulPartitionedCall2N
%dense_1_nn_36/StatefulPartitionedCall%dense_1_nn_36/StatefulPartitionedCall2N
%dense_2_nn_36/StatefulPartitionedCall%dense_2_nn_36/StatefulPartitionedCall2N
%dense_3_nn_36/StatefulPartitionedCall%dense_3_nn_36/StatefulPartitionedCall2N
%dense_4_nn_36/StatefulPartitionedCall%dense_4_nn_36/StatefulPartitionedCall2N
%dense_5_nn_36/StatefulPartitionedCall%dense_5_nn_36/StatefulPartitionedCall2N
%dense_6_nn_36/StatefulPartitionedCall%dense_6_nn_36/StatefulPartitionedCall2N
%dense_7_nn_36/StatefulPartitionedCall%dense_7_nn_36/StatefulPartitionedCall2\
,dense_out_nl_8_nn_36/StatefulPartitionedCall,dense_out_nl_8_nn_36/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:e a
0
_output_shapes
:������������������
-
_user_specified_namenormalization_input
�`
�	
G__inference_sequential_layer_call_and_return_conditional_losses_2193376

inputs1
-normalization_reshape_readvariableop_resource3
/normalization_reshape_1_readvariableop_resource
dense_0_nn_36_2193322
dense_0_nn_36_2193324
dense_1_nn_36_2193328
dense_1_nn_36_2193330
dense_2_nn_36_2193334
dense_2_nn_36_2193336
dense_3_nn_36_2193340
dense_3_nn_36_2193342
dense_4_nn_36_2193346
dense_4_nn_36_2193348
dense_5_nn_36_2193352
dense_5_nn_36_2193354
dense_6_nn_36_2193358
dense_6_nn_36_2193360
dense_7_nn_36_2193364
dense_7_nn_36_2193366 
dense_out_nl_8_nn_36_2193370 
dense_out_nl_8_nn_36_2193372
identity��%dense_0_nn_36/StatefulPartitionedCall�%dense_1_nn_36/StatefulPartitionedCall�%dense_2_nn_36/StatefulPartitionedCall�%dense_3_nn_36/StatefulPartitionedCall�%dense_4_nn_36/StatefulPartitionedCall�%dense_5_nn_36/StatefulPartitionedCall�%dense_6_nn_36/StatefulPartitionedCall�%dense_7_nn_36/StatefulPartitionedCall�,dense_out_nl_8_nn_36/StatefulPartitionedCall�$normalization/Reshape/ReadVariableOp�&normalization/Reshape_1/ReadVariableOp�
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp�
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape/shape�
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape�
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp�
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB"      2
normalization/Reshape_1/shape�
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*
_output_shapes

:2
normalization/Reshape_1�
normalization/subSubinputsnormalization/Reshape:output:0*
T0*'
_output_shapes
:���������2
normalization/sub{
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*
_output_shapes

:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *���32
normalization/Maximum/y�
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*
_output_shapes

:2
normalization/Maximum�
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*'
_output_shapes
:���������2
normalization/truediv�
%dense_0_nn_36/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0dense_0_nn_36_2193322dense_0_nn_36_2193324*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_21927892'
%dense_0_nn_36/StatefulPartitionedCall�
#leaky_re_lu_0_nn_36/PartitionedCallPartitionedCall.dense_0_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_21928102%
#leaky_re_lu_0_nn_36/PartitionedCall�
%dense_1_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_0_nn_36/PartitionedCall:output:0dense_1_nn_36_2193328dense_1_nn_36_2193330*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_21928282'
%dense_1_nn_36/StatefulPartitionedCall�
#leaky_re_lu_1_nn_36/PartitionedCallPartitionedCall.dense_1_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_21928492%
#leaky_re_lu_1_nn_36/PartitionedCall�
%dense_2_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_1_nn_36/PartitionedCall:output:0dense_2_nn_36_2193334dense_2_nn_36_2193336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_21928672'
%dense_2_nn_36/StatefulPartitionedCall�
#leaky_re_lu_2_nn_36/PartitionedCallPartitionedCall.dense_2_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_21928882%
#leaky_re_lu_2_nn_36/PartitionedCall�
%dense_3_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_2_nn_36/PartitionedCall:output:0dense_3_nn_36_2193340dense_3_nn_36_2193342*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_21929062'
%dense_3_nn_36/StatefulPartitionedCall�
#leaky_re_lu_3_nn_36/PartitionedCallPartitionedCall.dense_3_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_21929272%
#leaky_re_lu_3_nn_36/PartitionedCall�
%dense_4_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_3_nn_36/PartitionedCall:output:0dense_4_nn_36_2193346dense_4_nn_36_2193348*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_21929452'
%dense_4_nn_36/StatefulPartitionedCall�
#leaky_re_lu_4_nn_36/PartitionedCallPartitionedCall.dense_4_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_21929662%
#leaky_re_lu_4_nn_36/PartitionedCall�
%dense_5_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_4_nn_36/PartitionedCall:output:0dense_5_nn_36_2193352dense_5_nn_36_2193354*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_21929842'
%dense_5_nn_36/StatefulPartitionedCall�
#leaky_re_lu_5_nn_36/PartitionedCallPartitionedCall.dense_5_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_21930052%
#leaky_re_lu_5_nn_36/PartitionedCall�
%dense_6_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_5_nn_36/PartitionedCall:output:0dense_6_nn_36_2193358dense_6_nn_36_2193360*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_21930232'
%dense_6_nn_36/StatefulPartitionedCall�
#leaky_re_lu_6_nn_36/PartitionedCallPartitionedCall.dense_6_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_21930442%
#leaky_re_lu_6_nn_36/PartitionedCall�
%dense_7_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_6_nn_36/PartitionedCall:output:0dense_7_nn_36_2193364dense_7_nn_36_2193366*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *S
fNRL
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_21930622'
%dense_7_nn_36/StatefulPartitionedCall�
#leaky_re_lu_7_nn_36/PartitionedCallPartitionedCall.dense_7_nn_36/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_21930832%
#leaky_re_lu_7_nn_36/PartitionedCall�
,dense_out_nl_8_nn_36/StatefulPartitionedCallStatefulPartitionedCall,leaky_re_lu_7_nn_36/PartitionedCall:output:0dense_out_nl_8_nn_36_2193370dense_out_nl_8_nn_36_2193372*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Z
fURS
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_21931012.
,dense_out_nl_8_nn_36/StatefulPartitionedCall�
IdentityIdentity5dense_out_nl_8_nn_36/StatefulPartitionedCall:output:0&^dense_0_nn_36/StatefulPartitionedCall&^dense_1_nn_36/StatefulPartitionedCall&^dense_2_nn_36/StatefulPartitionedCall&^dense_3_nn_36/StatefulPartitionedCall&^dense_4_nn_36/StatefulPartitionedCall&^dense_5_nn_36/StatefulPartitionedCall&^dense_6_nn_36/StatefulPartitionedCall&^dense_7_nn_36/StatefulPartitionedCall-^dense_out_nl_8_nn_36/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*
_input_shapesn
l:������������������::::::::::::::::::::2N
%dense_0_nn_36/StatefulPartitionedCall%dense_0_nn_36/StatefulPartitionedCall2N
%dense_1_nn_36/StatefulPartitionedCall%dense_1_nn_36/StatefulPartitionedCall2N
%dense_2_nn_36/StatefulPartitionedCall%dense_2_nn_36/StatefulPartitionedCall2N
%dense_3_nn_36/StatefulPartitionedCall%dense_3_nn_36/StatefulPartitionedCall2N
%dense_4_nn_36/StatefulPartitionedCall%dense_4_nn_36/StatefulPartitionedCall2N
%dense_5_nn_36/StatefulPartitionedCall%dense_5_nn_36/StatefulPartitionedCall2N
%dense_6_nn_36/StatefulPartitionedCall%dense_6_nn_36/StatefulPartitionedCall2N
%dense_7_nn_36/StatefulPartitionedCall%dense_7_nn_36/StatefulPartitionedCall2\
,dense_out_nl_8_nn_36/StatefulPartitionedCall,dense_out_nl_8_nn_36/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp:X T
0
_output_shapes
:������������������
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_2192888

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
Q
5__inference_leaky_re_lu_4_nn_36_layer_call_fn_2193867

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������$* 
_read_only_resource_inputs
 *6
config_proto&$

CPU

GPU2*0,1,2,3J 8� *Y
fTRR
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_21929662
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_2193906

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�
l
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_2193862

inputs
identityd
	LeakyRelu	LeakyReluinputs*'
_output_shapes
:���������$*
alpha%���=2
	LeakyReluk
IdentityIdentityLeakyRelu:activations:0*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*&
_input_shapes
:���������$:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs
�	
�
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_2193935

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:$$*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:$*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������$2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������$2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������$::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:���������$
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
\
normalization_inputE
%serving_default_normalization_input:0������������������H
dense_out_nl_8_nn_360
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�a
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer-4
layer_with_weights-3
layer-5
layer-6
layer_with_weights-4
layer-7
	layer-8

layer_with_weights-5

layer-9
layer-10
layer_with_weights-6
layer-11
layer-12
layer_with_weights-7
layer-13
layer-14
layer_with_weights-8
layer-15
layer-16
layer_with_weights-9
layer-17
	optimizer
regularization_losses
	variables
trainable_variables
	keras_api

signatures
+�&call_and_return_all_conditional_losses
�_default_save_signature
�__call__"�[
_tf_keras_sequential�[{"class_name": "Sequential", "name": "sequential", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_input"}}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_0_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_0_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_1_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_2_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_3_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_4_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_5_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_6_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_7_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_out_nl_8_nn_36", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, null]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "sequential", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "normalization_input"}}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "Dense", "config": {"name": "dense_0_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_0_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_1_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_1_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_2_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_2_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_3_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_3_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_4_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_4_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_5_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_5_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_6_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_6_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_7_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "LeakyReLU", "config": {"name": "leaky_re_lu_7_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}, {"class_name": "Dense", "config": {"name": "dense_out_nl_8_nn_36", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": "mean_absolute_error", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
�
state_variables
_broadcast_shape
mean
variance
	count
	keras_api"�
_tf_keras_layer�{"class_name": "Normalization", "name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "stateful": false, "must_restore_from_config": true, "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "build_input_shape": [512, 13]}
�

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_0_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_0_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 13}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13]}}
�
%regularization_losses
&	variables
'trainable_variables
(	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_0_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_0_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

)kernel
*bias
+regularization_losses
,	variables
-trainable_variables
.	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_1_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
/regularization_losses
0	variables
1trainable_variables
2	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_1_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_1_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

3kernel
4bias
5regularization_losses
6	variables
7trainable_variables
8	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_2_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
9regularization_losses
:	variables
;trainable_variables
<	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_2_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_2_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

=kernel
>bias
?regularization_losses
@	variables
Atrainable_variables
B	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_3_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
Cregularization_losses
D	variables
Etrainable_variables
F	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_3_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_3_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

Gkernel
Hbias
Iregularization_losses
J	variables
Ktrainable_variables
L	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_4_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
Mregularization_losses
N	variables
Otrainable_variables
P	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_4_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_4_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

Qkernel
Rbias
Sregularization_losses
T	variables
Utrainable_variables
V	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_5_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
Wregularization_losses
X	variables
Ytrainable_variables
Z	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_5_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_5_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

[kernel
\bias
]regularization_losses
^	variables
_trainable_variables
`	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_6_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
aregularization_losses
b	variables
ctrainable_variables
d	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_6_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_6_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

ekernel
fbias
gregularization_losses
h	variables
itrainable_variables
j	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_7_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7_nn_36", "trainable": true, "dtype": "float32", "units": 36, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
kregularization_losses
l	variables
mtrainable_variables
n	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "LeakyReLU", "name": "leaky_re_lu_7_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "leaky_re_lu_7_nn_36", "trainable": true, "dtype": "float32", "alpha": 0.10000000149011612}}
�

okernel
pbias
qregularization_losses
r	variables
strainable_variables
t	keras_api
+�&call_and_return_all_conditional_losses
�__call__"�
_tf_keras_layer�{"class_name": "Dense", "name": "dense_out_nl_8_nn_36", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_out_nl_8_nn_36", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 36}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 36]}}
�
uiter

vbeta_1

wbeta_2
	xdecay
ylearning_ratem� m�)m�*m�3m�4m�=m�>m�Gm�Hm�Qm�Rm�[m�\m�em�fm�om�pm�v� v�)v�*v�3v�4v�=v�>v�Gv�Hv�Qv�Rv�[v�\v�ev�fv�ov�pv�"
	optimizer
 "
trackable_list_wrapper
�
0
1
2
3
 4
)5
*6
37
48
=9
>10
G11
H12
Q13
R14
[15
\16
e17
f18
o19
p20"
trackable_list_wrapper
�
0
 1
)2
*3
34
45
=6
>7
G8
H9
Q10
R11
[12
\13
e14
f15
o16
p17"
trackable_list_wrapper
�
regularization_losses
zmetrics
{layer_metrics
|non_trainable_variables
	variables

}layers
trainable_variables
~layer_regularization_losses
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
C
mean
variance
	count"
trackable_dict_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
&:$$2dense_0_nn_36/kernel
 :$2dense_0_nn_36/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
�
!regularization_losses
metrics
�layer_metrics
�non_trainable_variables
"	variables
�layers
#trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
%regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
&	variables
�layers
'trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_1_nn_36/kernel
 :$2dense_1_nn_36/bias
 "
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
.
)0
*1"
trackable_list_wrapper
�
+regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
,	variables
�layers
-trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
/regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
0	variables
�layers
1trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_2_nn_36/kernel
 :$2dense_2_nn_36/bias
 "
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
�
5regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
6	variables
�layers
7trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
9regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
:	variables
�layers
;trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_3_nn_36/kernel
 :$2dense_3_nn_36/bias
 "
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
?regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
@	variables
�layers
Atrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Cregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
D	variables
�layers
Etrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_4_nn_36/kernel
 :$2dense_4_nn_36/bias
 "
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
.
G0
H1"
trackable_list_wrapper
�
Iregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
J	variables
�layers
Ktrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Mregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
N	variables
�layers
Otrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_5_nn_36/kernel
 :$2dense_5_nn_36/bias
 "
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
�
Sregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
T	variables
�layers
Utrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
Wregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
X	variables
�layers
Ytrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_6_nn_36/kernel
 :$2dense_6_nn_36/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
�
]regularization_losses
�metrics
�layer_metrics
�non_trainable_variables
^	variables
�layers
_trainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
aregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
b	variables
�layers
ctrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
&:$$$2dense_7_nn_36/kernel
 :$2dense_7_nn_36/bias
 "
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
.
e0
f1"
trackable_list_wrapper
�
gregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
h	variables
�layers
itrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
kregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
l	variables
�layers
mtrainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-:+$2dense_out_nl_8_nn_36/kernel
':%2dense_out_nl_8_nn_36/bias
 "
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
.
o0
p1"
trackable_list_wrapper
�
qregularization_losses
�metrics
�layer_metrics
�non_trainable_variables
r	variables
�layers
strainable_variables
 �layer_regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
(
�0"
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�

�total

�count
�	variables
�	keras_api"�
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
:  (2total
:  (2count
0
�0
�1"
trackable_list_wrapper
.
�	variables"
_generic_user_object
+:)$2Adam/dense_0_nn_36/kernel/m
%:#$2Adam/dense_0_nn_36/bias/m
+:)$$2Adam/dense_1_nn_36/kernel/m
%:#$2Adam/dense_1_nn_36/bias/m
+:)$$2Adam/dense_2_nn_36/kernel/m
%:#$2Adam/dense_2_nn_36/bias/m
+:)$$2Adam/dense_3_nn_36/kernel/m
%:#$2Adam/dense_3_nn_36/bias/m
+:)$$2Adam/dense_4_nn_36/kernel/m
%:#$2Adam/dense_4_nn_36/bias/m
+:)$$2Adam/dense_5_nn_36/kernel/m
%:#$2Adam/dense_5_nn_36/bias/m
+:)$$2Adam/dense_6_nn_36/kernel/m
%:#$2Adam/dense_6_nn_36/bias/m
+:)$$2Adam/dense_7_nn_36/kernel/m
%:#$2Adam/dense_7_nn_36/bias/m
2:0$2"Adam/dense_out_nl_8_nn_36/kernel/m
,:*2 Adam/dense_out_nl_8_nn_36/bias/m
+:)$2Adam/dense_0_nn_36/kernel/v
%:#$2Adam/dense_0_nn_36/bias/v
+:)$$2Adam/dense_1_nn_36/kernel/v
%:#$2Adam/dense_1_nn_36/bias/v
+:)$$2Adam/dense_2_nn_36/kernel/v
%:#$2Adam/dense_2_nn_36/bias/v
+:)$$2Adam/dense_3_nn_36/kernel/v
%:#$2Adam/dense_3_nn_36/bias/v
+:)$$2Adam/dense_4_nn_36/kernel/v
%:#$2Adam/dense_4_nn_36/bias/v
+:)$$2Adam/dense_5_nn_36/kernel/v
%:#$2Adam/dense_5_nn_36/bias/v
+:)$$2Adam/dense_6_nn_36/kernel/v
%:#$2Adam/dense_6_nn_36/bias/v
+:)$$2Adam/dense_7_nn_36/kernel/v
%:#$2Adam/dense_7_nn_36/bias/v
2:0$2"Adam/dense_out_nl_8_nn_36/kernel/v
,:*2 Adam/dense_out_nl_8_nn_36/bias/v
�2�
G__inference_sequential_layer_call_and_return_conditional_losses_2193118
G__inference_sequential_layer_call_and_return_conditional_losses_2193632
G__inference_sequential_layer_call_and_return_conditional_losses_2193188
G__inference_sequential_layer_call_and_return_conditional_losses_2193553�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
"__inference__wrapped_model_2192762�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *;�8
6�3
normalization_input������������������
�2�
,__inference_sequential_layer_call_fn_2193304
,__inference_sequential_layer_call_fn_2193722
,__inference_sequential_layer_call_fn_2193677
,__inference_sequential_layer_call_fn_2193419�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_2193732�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_0_nn_36_layer_call_fn_2193741�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_2193746�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_0_nn_36_layer_call_fn_2193751�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_2193761�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_1_nn_36_layer_call_fn_2193770�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_2193775�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_1_nn_36_layer_call_fn_2193780�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_2193790�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_2_nn_36_layer_call_fn_2193799�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_2193804�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_2_nn_36_layer_call_fn_2193809�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_2193819�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_3_nn_36_layer_call_fn_2193828�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_2193833�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_3_nn_36_layer_call_fn_2193838�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_2193848�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_4_nn_36_layer_call_fn_2193857�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_2193862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_4_nn_36_layer_call_fn_2193867�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_2193877�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_5_nn_36_layer_call_fn_2193886�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_2193891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_5_nn_36_layer_call_fn_2193896�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_2193906�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_6_nn_36_layer_call_fn_2193915�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_2193920�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_6_nn_36_layer_call_fn_2193925�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_2193935�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
/__inference_dense_7_nn_36_layer_call_fn_2193944�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_2193949�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_leaky_re_lu_7_nn_36_layer_call_fn_2193954�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_2193964�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
6__inference_dense_out_nl_8_nn_36_layer_call_fn_2193973�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
%__inference_signature_wrapper_2193474normalization_input"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
"__inference__wrapped_model_2192762� )*34=>GHQR[\efopE�B
;�8
6�3
normalization_input������������������
� "K�H
F
dense_out_nl_8_nn_36.�+
dense_out_nl_8_nn_36����������
J__inference_dense_0_nn_36_layer_call_and_return_conditional_losses_2193732\ /�,
%�"
 �
inputs���������
� "%�"
�
0���������$
� �
/__inference_dense_0_nn_36_layer_call_fn_2193741O /�,
%�"
 �
inputs���������
� "����������$�
J__inference_dense_1_nn_36_layer_call_and_return_conditional_losses_2193761\)*/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_1_nn_36_layer_call_fn_2193770O)*/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_2_nn_36_layer_call_and_return_conditional_losses_2193790\34/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_2_nn_36_layer_call_fn_2193799O34/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_3_nn_36_layer_call_and_return_conditional_losses_2193819\=>/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_3_nn_36_layer_call_fn_2193828O=>/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_4_nn_36_layer_call_and_return_conditional_losses_2193848\GH/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_4_nn_36_layer_call_fn_2193857OGH/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_5_nn_36_layer_call_and_return_conditional_losses_2193877\QR/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_5_nn_36_layer_call_fn_2193886OQR/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_6_nn_36_layer_call_and_return_conditional_losses_2193906\[\/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_6_nn_36_layer_call_fn_2193915O[\/�,
%�"
 �
inputs���������$
� "����������$�
J__inference_dense_7_nn_36_layer_call_and_return_conditional_losses_2193935\ef/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
/__inference_dense_7_nn_36_layer_call_fn_2193944Oef/�,
%�"
 �
inputs���������$
� "����������$�
Q__inference_dense_out_nl_8_nn_36_layer_call_and_return_conditional_losses_2193964\op/�,
%�"
 �
inputs���������$
� "%�"
�
0���������
� �
6__inference_dense_out_nl_8_nn_36_layer_call_fn_2193973Oop/�,
%�"
 �
inputs���������$
� "�����������
P__inference_leaky_re_lu_0_nn_36_layer_call_and_return_conditional_losses_2193746X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_0_nn_36_layer_call_fn_2193751K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_1_nn_36_layer_call_and_return_conditional_losses_2193775X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_1_nn_36_layer_call_fn_2193780K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_2_nn_36_layer_call_and_return_conditional_losses_2193804X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_2_nn_36_layer_call_fn_2193809K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_3_nn_36_layer_call_and_return_conditional_losses_2193833X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_3_nn_36_layer_call_fn_2193838K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_4_nn_36_layer_call_and_return_conditional_losses_2193862X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_4_nn_36_layer_call_fn_2193867K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_5_nn_36_layer_call_and_return_conditional_losses_2193891X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_5_nn_36_layer_call_fn_2193896K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_6_nn_36_layer_call_and_return_conditional_losses_2193920X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_6_nn_36_layer_call_fn_2193925K/�,
%�"
 �
inputs���������$
� "����������$�
P__inference_leaky_re_lu_7_nn_36_layer_call_and_return_conditional_losses_2193949X/�,
%�"
 �
inputs���������$
� "%�"
�
0���������$
� �
5__inference_leaky_re_lu_7_nn_36_layer_call_fn_2193954K/�,
%�"
 �
inputs���������$
� "����������$�
G__inference_sequential_layer_call_and_return_conditional_losses_2193118� )*34=>GHQR[\efopM�J
C�@
6�3
normalization_input������������������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_2193188� )*34=>GHQR[\efopM�J
C�@
6�3
normalization_input������������������
p 

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_2193553 )*34=>GHQR[\efop@�=
6�3
)�&
inputs������������������
p

 
� "%�"
�
0���������
� �
G__inference_sequential_layer_call_and_return_conditional_losses_2193632 )*34=>GHQR[\efop@�=
6�3
)�&
inputs������������������
p 

 
� "%�"
�
0���������
� �
,__inference_sequential_layer_call_fn_2193304 )*34=>GHQR[\efopM�J
C�@
6�3
normalization_input������������������
p

 
� "�����������
,__inference_sequential_layer_call_fn_2193419 )*34=>GHQR[\efopM�J
C�@
6�3
normalization_input������������������
p 

 
� "�����������
,__inference_sequential_layer_call_fn_2193677r )*34=>GHQR[\efop@�=
6�3
)�&
inputs������������������
p

 
� "�����������
,__inference_sequential_layer_call_fn_2193722r )*34=>GHQR[\efop@�=
6�3
)�&
inputs������������������
p 

 
� "�����������
%__inference_signature_wrapper_2193474� )*34=>GHQR[\efop\�Y
� 
R�O
M
normalization_input6�3
normalization_input������������������"K�H
F
dense_out_nl_8_nn_36.�+
dense_out_nl_8_nn_36���������