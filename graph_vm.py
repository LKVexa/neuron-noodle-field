# SPDX-License-Identifier: GPL-3.0-only
"""Bounded tensor primitives for operator graphs carried in image pixels.

There are no OR/AND/neighborhood rule names in the executor. Offset choices,
weights, bias, constants, arithmetic, activation and threshold come from data.
"""
import copy
import math

import numpy as np


class GraphError(ValueError):
    pass


ARITIES = {"STATE":2,"PARAM":3,"CONST":3,"GATHER":4,"DOT":4,"ADD":4,
           "MUL":4,"SIGMOID":3,"GE":4,"RETURN":3}


def finite_number(value):
    if type(value) not in (int,float) or abs(value)>1e6 or not math.isfinite(value):
        raise GraphError("Graph numbers must be finite and bounded by 1000000")
    return float(value)


def validate_model(model):
    if type(model) is not dict or set(model)!={"weights","bias"}:
        raise GraphError("Model requires weights and bias")
    weights=model['weights']
    if type(weights) is not list or not 1<=len(weights)<=9:
        raise GraphError("Model requires 1..9 weights")
    for value in weights: finite_number(value)
    finite_number(model['bias'])
    return copy.deepcopy(model)


def validate_program(program):
    if type(program) is not dict or set(program)!={"schema","code"} or program['schema']!='nnf-graph/1':
        raise GraphError("Unsupported operator graph")
    code=program['code']
    if type(code) is not list or not 1<=len(code)<=32:
        raise GraphError("Graph requires 1..32 instructions")
    assigned=set()
    for index,ins in enumerate(code):
        if type(ins) is not list or not ins or type(ins[0]) is not str or ins[0] not in ARITIES or len(ins)!=ARITIES[ins[0]]:
            raise GraphError("Unknown graph opcode or arity")
        op=ins[0]
        def register(r):
            if type(r) is not int or not 0<=r<16: raise GraphError("Graph register outside 0..15")
        if op=='RETURN':
            if index!=len(code)-1: raise GraphError("RETURN must be the last instruction")
            sources=ins[1:]
        else:
            register(ins[1])
            if op=='STATE': sources=[]
            elif op=='PARAM':
                if ins[2] not in ('weights','bias'): raise GraphError("Unknown model parameter")
                sources=[]
            elif op=='CONST': finite_number(ins[2]); sources=[]
            elif op=='GATHER':
                offsets=ins[3]
                if type(offsets) is not list or not 1<=len(offsets)<=9: raise GraphError("GATHER requires 1..9 offsets")
                for pair in offsets:
                    if type(pair) is not list or len(pair)!=2 or any(type(x) is not int or abs(x)>2 for x in pair):
                        raise GraphError("GATHER offsets must be integers in -2..2")
                sources=[ins[2]]
            else: sources=ins[2:]
        for source in sources:
            register(source)
            if source not in assigned: raise GraphError("Uninitialized graph register")
        if op!='RETURN': assigned.add(ins[1])
    if code[-1][0]!='RETURN': raise GraphError("Graph must end with RETURN")
    return copy.deepcopy(program)


def validate_state(state):
    if type(state) is not np.ndarray or state.dtype!=np.uint8 or state.ndim!=2 or state.shape[0]!=state.shape[1] or not 8<=state.shape[0]<=64:
        raise GraphError("State requires a square uint8 array of size 8..64")
    if not np.all((state==0)|(state==1)): raise GraphError("State must be binary")


def _bounded(value,n):
    if np.isscalar(value):
        if not math.isfinite(float(value)) or abs(float(value))>1e13: raise GraphError("Intermediate scalar range exceeded")
    elif type(value) is np.ndarray:
        if value.shape not in [(n,n)] + [(n,n,k) for k in range(1,10)] + [(k,) for k in range(1,10)]:
            raise GraphError("Intermediate tensor shape exceeded")
        if not np.isfinite(value).all() or np.max(np.abs(value))>1e13: raise GraphError("Intermediate tensor range exceeded")
    else: raise GraphError("Unsupported graph value")
    return value


def execute(program,model,state):
    """Evaluate the image's graph with NumPy, without a named field rule."""
    program,model=validate_program(program),validate_model(model)
    validate_state(state)
    n=len(state); registers={}
    try:
        with np.errstate(over='raise',invalid='raise',divide='raise'):
            for ins in program['code']:
                op=ins[0]
                if op=='RETURN':
                    candidate,prob=registers[ins[1]],registers[ins[2]]
                    if not isinstance(candidate,np.ndarray) or candidate.shape!=state.shape or not np.all((candidate==0)|(candidate==1)):
                        raise GraphError("RETURN state must be a binary grid")
                    if not isinstance(prob,np.ndarray) or prob.shape!=state.shape or not np.all((prob>=0)&(prob<=1)):
                        raise GraphError("RETURN probability must be a grid in [0,1]")
                    return candidate.astype(np.uint8),prob.astype(np.float64)
                if op=='STATE': value=state.astype(np.float64)
                elif op=='PARAM': value=np.asarray(model['weights'],dtype=np.float64) if ins[2]=='weights' else float(model['bias'])
                elif op=='CONST': value=float(ins[2])
                elif op=='GATHER':
                    source=registers[ins[2]]
                    if not isinstance(source,np.ndarray) or source.shape!=(n,n): raise GraphError("GATHER requires a grid")
                    padded=np.pad(source,2,constant_values=0)
                    value=np.stack([padded[2+dy:2+dy+n,2+dx:2+dx+n] for dy,dx in ins[3]],axis=-1)
                elif op=='DOT':
                    features,weights=registers[ins[2]],registers[ins[3]]
                    if not isinstance(features,np.ndarray) or features.ndim!=3 or not isinstance(weights,np.ndarray) or weights.ndim!=1 or features.shape[2]!=weights.size:
                        raise GraphError("DOT feature/model dimension mismatch")
                    value=features@weights
                elif op in ('ADD','MUL','GE'):
                    a,b=registers[ins[2]],registers[ins[3]]
                    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray) and a.shape!=b.shape: raise GraphError("Binary tensor shapes must match")
                    value=a+b if op=='ADD' else (a*b if op=='MUL' else a>=b)
                elif op=='SIGMOID': value=1/(1+np.exp(-np.clip(registers[ins[2]],-60,60)))
                registers[ins[1]]=_bounded(value,n)
    except (ValueError,TypeError,KeyError,OverflowError,FloatingPointError) as exc:
        if isinstance(exc,GraphError): raise
        raise GraphError("Invalid graph tensor operation") from exc
    raise GraphError("Graph did not RETURN")


def execute_reference(program,model,state):
    """Independent Python-list/scalar implementation of the same primitives."""
    program,model=validate_program(program),validate_model(model)
    validate_state(state)
    n=len(state); registers={}
    def scalar(value): return type(value) in (int,float,bool)
    def unary(value,fn):
        return fn(value) if scalar(value) else [unary(v,fn) for v in value]
    def binary(a,b,fn):
        if scalar(a) and scalar(b): return fn(a,b)
        if scalar(a): return [binary(a,v,fn) for v in b]
        if scalar(b): return [binary(v,b,fn) for v in a]
        if len(a)!=len(b): raise GraphError("Reference shape mismatch")
        return [binary(x,y,fn) for x,y in zip(a,b)]
    def probability(value): return 1/(1+math.exp(-max(-60,min(60,value))))
    for ins in program['code']:
        op=ins[0]
        if op=='RETURN':
            candidate=np.asarray(registers[ins[1]],dtype=np.float64)
            probabilities=np.asarray(registers[ins[2]],dtype=np.float64)
            if candidate.shape!=state.shape or not np.all((candidate==0)|(candidate==1)):
                raise GraphError("Reference RETURN state must be a binary grid")
            if probabilities.shape!=state.shape or not np.all((probabilities>=0)&(probabilities<=1)):
                raise GraphError("Reference RETURN probability must be a grid in [0,1]")
            return candidate.astype(np.uint8),probabilities
        if op=='STATE': value=state.tolist()
        elif op=='PARAM': value=copy.deepcopy(model[ins[2]])
        elif op=='CONST': value=ins[2]
        elif op=='GATHER':
            source=registers[ins[2]]
            if np.asarray(source).shape!=(n,n): raise GraphError("Reference GATHER requires a grid")
            value=[[[source[y+dy][x+dx] if 0<=y+dy<n and 0<=x+dx<n else 0 for dy,dx in ins[3]] for x in range(n)] for y in range(n)]
        elif op=='DOT':
            source,weights=registers[ins[2]],registers[ins[3]]
            shape=np.asarray(source).shape
            if len(shape)!=3 or not isinstance(weights,list) or shape[-1]!=len(weights) or any(not scalar(v) for v in weights):
                raise GraphError("Reference DOT feature/model dimension mismatch")
            value=[[math.fsum(float(v)*float(w) for v,w in zip(cell,weights)) for cell in row] for row in source]
        elif op=='ADD': value=binary(registers[ins[2]],registers[ins[3]],lambda a,b:a+b)
        elif op=='MUL': value=binary(registers[ins[2]],registers[ins[3]],lambda a,b:a*b)
        elif op=='GE': value=binary(registers[ins[2]],registers[ins[3]],lambda a,b:a>=b)
        elif op=='SIGMOID': value=unary(registers[ins[2]],probability)
        # Bound the reference entry point too, before an integer can be squared
        # repeatedly. NumPy is used here only to inspect shape and finite range;
        # all reference arithmetic above remains scalar/list based.
        _bounded(float(value) if scalar(value) else np.asarray(value,dtype=np.float64),n)
        registers[ins[1]]=value
    raise GraphError("Reference graph did not RETURN")


def checked_step(program,model,state):
    candidate,probabilities=execute(program,model,state)
    expected,reference_probabilities=execute_reference(program,model,state)
    if not np.array_equal(candidate,expected) or not np.allclose(probabilities,reference_probabilities,rtol=1e-12,atol=1e-12):
        raise GraphError("Image graph result failed independent scalar validation")
    return candidate.copy(),probabilities.copy()
